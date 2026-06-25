import io
import json
import logging
import threading
import time

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from skimage.segmentation import slic

import os
import urllib.request

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator as SamMaskGenerator
from mobile_sam import sam_model_registry as mobile_sam_registry, SamAutomaticMaskGenerator as MobileSamMaskGenerator

from server import PromptServer  # type: ignore
from aiohttp import web as _aiohttp_web

_SERVER_AVAILABLE = True

_sam_choice_event = threading.Event()
_sam_choice_result: Dict[str, Any] = {}


@PromptServer.instance.routes.post("/artishow/sam_download_choice")
async def _sam_choice_handler(request):
    global _sam_choice_result
    _sam_choice_result = await request.json()
    _sam_choice_event.set()
    return _aiohttp_web.json_response({"status": "ok"})


log = logging.getLogger(__name__)

_BOUNDARY_COLOUR: Tuple[int, int, int, int] = (255, 80, 0, 220)
_CACHE_TTL: int = 600  # 10 minutes

_label_map_cache: Dict[Tuple, Dict[str, Any]] = {}
_cache_lock = threading.Lock()

SAM_MODELS = {
    "mobile_sam.pt": {
        "type": "mobile_sam",
        "url": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        "size": "39 MB",
    },
    "sam_vit_b_01ec64.pth": {
        "type": "vit_b",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "size": "375 MB",
    },
    "sam_vit_l_0b3195.pth": {
        "type": "vit_l",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "size": "1.25 GB",
    },
    "sam_vit_h_4b8939.pth": {
        "type": "vit_h",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "size": "2.56 GB",
    },
}


def _get_checkpoint_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, "models/sams/")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def _find_available_sam_models(checkpoint_dir: str) -> List[Dict[str, str]]:
    """
    Scan checkpoint_dir and return a list of recognised SAM model files found.
    Each entry: {"filename": ..., "type": ..., "path": ...}
    """
    found = []
    for filename, meta in SAM_MODELS.items():
        path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(path):
            found.append(
                {
                    "filename": filename,
                    "type": meta["type"],
                    "path": path,
                }
            )
    return found


def _prompt_user_download_sam(checkpoint_dir: str) -> Dict[str, str]:
    """
    Send a dialog event to the ComfyUI frontend asking the user to choose
    which SAM model to download. Blocks until the user replies.

    Returns the chosen model dict from SAM_MODELS, or raises RuntimeError
    if the user cancels or the server is unavailable.
    """
    if not _SERVER_AVAILABLE:
        raise RuntimeError(
            "No SAM model found and PromptServer unavailable — "
            "please manually place a SAM checkpoint in: " + checkpoint_dir
        )

    choices = [
        {
            "filename": filename,
            "label": f"{filename}  ({meta['size']})",
            "type": meta["type"],
            "url": meta["url"],
        }
        for filename, meta in SAM_MODELS.items()
    ]

    # Reset avant d'envoyer l'event
    _sam_choice_event.clear()
    _sam_choice_result.clear()

    PromptServer.instance.send_sync(
        "sam_model_missing",
        {
            "checkpoint_dir": checkpoint_dir,
            "choices": choices,
        },
    )

    log.info("Waiting for user SAM model selection …")
    if not _sam_choice_event.wait(timeout=120):
        raise RuntimeError("Timed out waiting for user to select a SAM model.")

    if _sam_choice_result.get("cancelled"):
        raise RuntimeError("User cancelled SAM model download.")

    chosen = _sam_choice_result.get("filename")
    if chosen not in SAM_MODELS:
        raise RuntimeError(f"Unknown model selected: {chosen!r}")

    return {"filename": chosen, **SAM_MODELS[chosen]}


def _download_sam_model(filename: str, url: str, checkpoint_dir: str) -> str:
    """Download a SAM checkpoint and return its local path."""
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    log.info("Downloading SAM checkpoint: %s …", url)

    try:
        urllib.request.urlretrieve(url, checkpoint_path)
    except Exception as e:
        # Clean up partial download
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        raise RuntimeError(f"Download failed: {e}") from e

    log.info("SAM checkpoint saved to %s", checkpoint_path)
    return checkpoint_path


def _evict_stale_cache() -> None:
    """Remove entries older than _CACHE_TTL.  Called lazily on each write."""
    now = time.time()
    stale = [
        k for k, v in _label_map_cache.items() if now - v["timestamp"] > _CACHE_TTL
    ]
    for k in stale:
        del _label_map_cache[k]
        log.debug("Evicted stale label_map cache for key=%s", k)


def _broadcast_segment_data(
    node_id: str,
    overlay_b64: str,
    id_map_b64: str,
    label_map: np.ndarray,
) -> None:
    """
    Send segment data to the frontend via WebSocket using ComfyUI's PromptServer.

    The frontend listens for 'interactive_segmask' events and updates
    the node preview with the new images.

    Parameters
    ----------
    node_id : str
        Unique node identifier.
    overlay_b64 : str
        Base-64 encoded overlay PNG.
    id_map_b64 : str
        Base-64 encoded ID-map PNG.
    label_map : np.ndarray
        [H, W] int32 label map (used to compute segment count).
    """
    if not _SERVER_AVAILABLE:
        log.warning(
            "ComfyUI PromptServer not available — WebSocket broadcast skipped. "
            "This is expected during isolated testing."
        )
        return

    try:
        server = PromptServer.instance  # type: ignore
        payload = {
            "node_id": node_id,
            "overlay_b64": overlay_b64,
            "id_map_b64": id_map_b64,
            "num_segments": int(label_map.max()) + 1,
            "width": label_map.shape[1],
            "height": label_map.shape[0],
        }
        # Broadcast to all connected clients
        server.send_sync("interactive_segmask", payload)  # type: ignore
        log.debug("WebSocket broadcast sent for node_id=%s", node_id)
    except Exception as exc:
        log.error("Failed to broadcast segment data via WebSocket: %s", exc)


def _tensor_to_numpy_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a ComfyUI image tensor [B, H, W, C] float32 [0, 1]
    to a uint8 numpy array [H, W, C] for the first batch item.
    """
    if image_tensor.ndim == 4:
        frame = image_tensor[0]  # take first batch item → [H, W, C]
    elif image_tensor.ndim == 3:
        frame = image_tensor
    else:
        raise ValueError(f"Unexpected image tensor shape: {image_tensor.shape}")

    arr = (frame.cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return arr  # [H, W, C]


def _numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Wrap a uint8 numpy [H, W, C] or [H, W] array in a Pillow Image."""
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA")
    return Image.fromarray(arr, mode="RGB")


def _pil_to_base64_png(img: Image.Image) -> str:
    """Encode a Pillow image as a base-64 PNG string (no data-URI prefix)."""
    import base64

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _deterministic_colour(segment_id: int) -> Tuple[int, int, int]:
    """
    Map an integer segment ID to a unique, visually distinct RGB colour.

    We use a simple golden-ratio based hue spread converted to RGB so that
    adjacent IDs get maximally different hues.  The colour is fully
    deterministic — the same ID always yields the same colour — which is
    essential for the JS hover detection logic.

    Segment ID 0 is reserved as "background / no segment" and maps to
    pure black (0, 0, 0).
    """
    if segment_id == 0:
        return (0, 0, 0)

    # Golden-ratio hue stepping keeps consecutive IDs visually distinct.
    golden = 0.618033988749895
    hue = (segment_id * golden) % 1.0

    # Convert HSV → RGB (full saturation, high value for easy JS pixel reads)
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.85)
    return (int(r * 255), int(g * 255), int(b * 255))


def _image_tensor_hash(tensor: torch.Tensor) -> int:
    """
    Produce a cheap hash for a tensor so we can detect image changes
    without storing the full tensor.  We sample a small subset of values.
    """
    flat = tensor.reshape(-1)
    step = max(1, flat.numel() // 512)
    sampled = flat[::step].cpu().float()
    return hash(tuple(sampled.tolist()))


def _run_slic_segmentation(image_rgb: np.ndarray, **kwargs) -> np.ndarray:
    """
    Run SLIC superpixel segmentation on *image_rgb* and return a label map.

    Parameters
    ----------
    image_rgb : np.ndarray
        uint8 array shaped [H, W, 3].
    **kwargs
        Additional hyper-parameters forwarded from the node widget values
        (e.g. n_segments, compactness, sigma).

    Returns
    -------
    label_map : np.ndarray
        int32 array shaped [H, W].  Every pixel carries its segment ID
        in the range [0, N-1] where N is the number of segments found.
    """
    label_map = slic(
        image_rgb,
        n_segments=kwargs.get("n_segments", 400),
        compactness=kwargs.get("compactness", 5.0),
        sigma=kwargs.get("sigma", 1.0),
        start_label=0,
        enforce_connectivity=True,
    )

    return label_map.astype(np.int32)


def _run_sam_segmentation(image_rgb: np.ndarray, **kwargs) -> np.ndarray:
    H, W = image_rgb.shape[:2]
    checkpoint_dir = _get_checkpoint_dir()

    # 1. Chercher les modèles déjà présents
    available = _find_available_sam_models(checkpoint_dir)

    if available:
        # Prendre le premier trouvé (priorité : vit_b > vit_l > vit_h selon l'ordre du dict)
        chosen = available[0]
        checkpoint_path = chosen["path"]
        model_type = chosen["type"]
        log.info("Using existing SAM model: %s (%s)", chosen["filename"], model_type)
    else:
        # 2. Aucun modèle trouvé → dialogue utilisateur
        log.warning("No SAM model found in %s", checkpoint_dir)
        try:
            model_info = _prompt_user_download_sam(checkpoint_dir)
        except RuntimeError as e:
            log.error("SAM model selection failed: %s", e)
            raise ValueError(str(e))

        checkpoint_path = _download_sam_model(
            filename=model_info["filename"],
            url=model_info["url"],
            checkpoint_dir=checkpoint_dir,
        )
        model_type = model_info["type"]

    # 3. Charger et exécuter SAM (inchangé)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if model_type == "mobile_sam":
            sam = mobile_sam_registry["vit_t"](checkpoint=checkpoint_path)
            MaskGenerator = MobileSamMaskGenerator
        else:
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            MaskGenerator = SamMaskGenerator

        sam.to(device=device)

        # Retrieval of hyperparameters from kwargs or optimal default values
        points_per_side = kwargs.get(
            "points_per_side", 32
        )  # Increase for more small segments
        pred_iou_thresh = kwargs.get("pred_iou_thresh", 0.88)
        stability_score_thresh = kwargs.get("stability_score_thresh", 0.95)

        generator = MaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Filter noise residuals smaller than 100 pixels
        )

    except Exception as e:
        log.error(f"Error while initializing SAM: {e}")
        raise ValueError(
            "Failed to initialize SAM model. Check if the checkpoint is valid."
        )

    masks = generator.generate(image_rgb)
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)

    label_map = np.zeros((H, W), dtype=np.int32)

    for idx, mask_dict in enumerate(masks, start=1):
        boolean_mask = mask_dict["segmentation"]
        label_map[boolean_mask] = idx

    return label_map


class InteractiveSegmentationMask:
    CATEGORY = "image/masking"
    FUNCTION = "execute"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "segmentation_engine": (
                    ["SLIC", "SAM"],
                    {"default": "SLIC"},
                ),
                "selected_coords": ("STRING", {"default": "[]"}),
                # SLIC hyperparameters
                "slic_n_segments": (
                    "INT",
                    {
                        "default": 400,
                        "min": 10,
                        "max": 2000,
                        "step": 10,
                        "display": "slider",
                    },
                ),
                "slic_compactness": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.1,
                        "max": 100.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "slic_sigma": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                # SAM hyperparameters
                "sam_points_per_side": (
                    "INT",
                    {
                        "default": 32,
                        "min": 8,
                        "max": 128,
                        "step": 4,
                        "display": "slider",
                    },
                ),
                "sam_pred_iou_thresh": (
                    "FLOAT",
                    {
                        "default": 0.88,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "sam_stability_score_thresh": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Indique à ComfyUI si le nœud a changé.
        En renvoyant la chaîne JSON des coordonnées, ComfyUI va comparer
        cette chaîne d'une exécution à l'autre. Le moindre clic ou dé-sélection
        invalidera le cache et forcera la réexécution du nœud.
        """
        return (
            kwargs.get("segmentation_engine", ""),
            kwargs.get("slic_n_segments", 400),
            kwargs.get("slic_compactness", 5.0),
            kwargs.get("slic_sigma", 1.0),
            kwargs.get("sam_points_per_side", 32),
            kwargs.get("sam_pred_iou_thresh", 0.88),
            kwargs.get("sam_stability_score_thresh", 0.95),
            kwargs.get("selected_coords", ""),
        )

    def execute(
        self,
        image: torch.Tensor,
        segmentation_engine: str,
        slic_n_segments: int = 400,
        slic_compactness: float = 5.0,
        slic_sigma: float = 1.0,
        sam_points_per_side: int = 32,
        sam_pred_iou_thresh: float = 0.88,
        sam_stability_score_thresh: float = 0.95,
        selected_coords: str = "[]",
        unique_id: str = "0",
    ) -> Tuple[torch.Tensor]:
        """
        Execute the node.

        Parameters

        image : torch.Tensor
            Shape [B, H, W, C], dtype float32, range [0, 1].
        segmentation_engine : str
            "SLIC" or "SAM".
        selected_coords : str
            JSON array of {x, y} objects representing user-clicked pixels.
        unique_id : str
            ComfyUI node UID — used as cache key.

        Returns

        Tuple containing a single MASK tensor of shape [1, H, W], float32,
        values in {0.0, 1.0}.
        """
        log.info(
            "InteractiveSegmentationMask.execute: engine=%s, node_id=%s",
            segmentation_engine,
            unique_id,
        )

        image_np = _tensor_to_numpy_uint8(image)  # [H, W, C]
        H, W = image_np.shape[:2]
        image_hash = _image_tensor_hash(image)

        engine_kwargs = (
            {
                "n_segments": slic_n_segments,
                "compactness": slic_compactness,
                "sigma": slic_sigma,
            }
            if segmentation_engine == "SLIC"
            else {
                "points_per_side": sam_points_per_side,
                "pred_iou_thresh": sam_pred_iou_thresh,
                "stability_score_thresh": sam_stability_score_thresh,
            }
        )

        label_map = self._get_or_compute_segments(
            image_np=image_np,
            engine=segmentation_engine,
            node_id=unique_id,
            image_hash=image_hash,
            engine_kwargs=engine_kwargs,
        )

        self._update_frontend_cache(
            node_id=unique_id,
            image_np=image_np,
            label_map=label_map,
            image_hash=image_hash,
        )

        mask_tensor = self._build_mask_from_selections(
            label_map=label_map,
            selected_coords_json=selected_coords,
            H=H,
            W=W,
        )

        return (mask_tensor,)

    def _get_or_compute_segments(
        self,
        image_np: np.ndarray,
        engine: str,
        node_id: str,
        image_hash: int,
        engine_kwargs: dict = {},
    ) -> np.ndarray:
        """
        Return a cached label map if the image hasn't changed; otherwise
        re-run the chosen segmentation engine.

        Parameters

        image_np : np.ndarray  [H, W, 3] uint8
        engine   : str         "SLIC" | "SAM"
        node_id  : str         Node identifier
        image_hash : int       Cheap hash of the source tensor

        Returns

        label_map : np.ndarray [H, W] int32
        """
        params_key = tuple(sorted(engine_kwargs.items()))
        cache_key = (node_id, image_hash, params_key)

        with _cache_lock:
            cached = _label_map_cache.get(cache_key)
            if cached is not None:
                log.debug("Label_map cache hit for node_id=%s", node_id)
                return cached["label_map"]

        log.info("Running %s segmentation for node_id=%s …", engine, node_id)

        # Only RGB is passed to the segmentation engines.
        image_rgb = image_np[..., :3] if image_np.shape[2] >= 3 else image_np

        if engine == "SLIC":
            label_map = _run_slic_segmentation(image_rgb, **engine_kwargs)
        elif engine == "SAM":
            label_map = _run_sam_segmentation(image_rgb, **engine_kwargs)
        else:
            raise ValueError(f"Unknown segmentation engine: {engine!r}")

        # Guarantee int32 and zero-indexed
        label_map = label_map.astype(np.int32)
        label_map = label_map - label_map.min()  # ensure 0-based

        log.info(
            "Segmentation complete: %d segments, image size %dx%d",
            label_map.max() + 1,
            label_map.shape[1],
            label_map.shape[0],
        )

        # Cache the label_map for future runs with the same image
        with _cache_lock:
            _label_map_cache[cache_key] = {
                "label_map": label_map,
                "timestamp": time.time(),
            }

        return label_map

    def _update_frontend_cache(
        self,
        node_id: str,
        image_np: np.ndarray,
        label_map: np.ndarray,
        image_hash: int,
    ) -> None:

        H, W = label_map.shape
        MAX_PREVIEW_SIZE = 512

        scale = min(MAX_PREVIEW_SIZE / W, MAX_PREVIEW_SIZE / H, 1.0)
        new_W, new_H = int(W * scale), int(H * scale)

        overlay_img = self._draw_boundary_overlay(image_np, label_map)
        if scale < 1.0:
            overlay_img = overlay_img.resize((new_W, new_H), Image.Resampling.LANCZOS)
        overlay_b64 = _pil_to_base64_png(overlay_img)

        id_map_img = self._draw_id_map(label_map)
        if scale < 1.0:
            id_map_img = id_map_img.resize((new_W, new_H), Image.Resampling.NEAREST)
        id_map_b64 = _pil_to_base64_png(id_map_img)

        _broadcast_segment_data(
            node_id=node_id,
            overlay_b64=overlay_b64,
            id_map_b64=id_map_b64,
            label_map=label_map,
        )
        log.debug("Frontend data broadcast for node_id=%s via WebSocket", node_id)

    def _draw_boundary_overlay(
        self, image_np: np.ndarray, label_map: np.ndarray
    ) -> Image.Image:
        """
        Produce an RGBA Pillow image: the original image with vivid boundary
        lines drawn where adjacent pixels belong to different segments.

        The boundary detection works by comparing each pixel with its right
        and bottom neighbours — no external library required.
        """
        H, W = label_map.shape

        # Detect horizontal and vertical edges
        h_edge = label_map[:-1, :] != label_map[1:, :]  # [H-1, W]
        v_edge = label_map[:, :-1] != label_map[:, 1:]  # [H, W-1]

        # Convert source image to RGBA
        pil_base = _numpy_to_pil(image_np).convert("RGBA")
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Draw horizontal edge lines
        hy, hx = np.where(h_edge)
        for y, x in zip(hy.tolist(), hx.tolist()):
            draw.line([(x, y + 1), (x + 1, y + 1)], fill=_BOUNDARY_COLOUR, width=1)

        # Draw vertical edge lines
        vy, vx = np.where(v_edge)
        for y, x in zip(vy.tolist(), vx.tolist()):
            draw.line([(x + 1, y), (x + 1, y + 1)], fill=_BOUNDARY_COLOUR, width=1)

        result = Image.alpha_composite(pil_base, overlay)
        return result

    def _draw_id_map(self, label_map: np.ndarray) -> Image.Image:
        """
        Produce an RGB Pillow image where every pixel is flood-filled with
        the deterministic colour corresponding to its segment ID.

        This image is never shown directly; it is loaded into an off-screen
        canvas by the JS frontend and queried via `getImageData` on mousemove
        to identify segments in O(1).
        """
        H, W = label_map.shape
        num_segments = label_map.max() + 1

        # Build a colour lookup table: segment_id → [R, G, B]
        lut = np.zeros((num_segments + 1, 3), dtype=np.uint8)
        for seg_id in range(num_segments):
            lut[seg_id] = _deterministic_colour(seg_id)

        # Vectorised lookup: shape [H, W, 3]
        id_map_rgb = lut[label_map]
        return _numpy_to_pil(id_map_rgb)

    def _build_mask_from_selections(
        self,
        label_map: np.ndarray,
        selected_coords_json: str,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Parse `selected_coords_json`, resolve each coordinate to its segment
        ID, union all matching pixels, and return a binary mask tensor.

        Parameters
        ----------
        label_map : np.ndarray  [H, W] int32
        selected_coords_json : str
            JSON array of objects: [{"x": int, "y": int}, …]
        H, W : int  Image dimensions

        Returns
        -------
        torch.Tensor  shape [1, H, W], dtype float32, values in {0.0, 1.0}
        """
        mask = np.zeros((H, W), dtype=np.float32)

        try:
            coords: List[Dict[str, int]] = json.loads(selected_coords_json or "[]")
        except json.JSONDecodeError as exc:
            log.warning(
                "Could not parse selected_coords JSON (%s) — returning empty mask. "
                "Raw value: %r",
                exc,
                selected_coords_json,
            )
            coords = []

        if not coords:
            log.debug("No selections — returning empty mask.")
            return torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        selected_segment_ids: set = set()
        for coord in coords:
            try:
                px = int(round(coord["x"]))
                py = int(round(coord["y"]))
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("Skipping malformed coordinate %r: %s", coord, exc)
                continue

            # Clamp to valid image bounds
            px = max(0, min(W - 1, px))
            py = max(0, min(H - 1, py))

            seg_id = int(label_map[py, px])
            selected_segment_ids.add(seg_id)
            log.debug("Coord (%d, %d) → segment_id=%d", px, py, seg_id)

        log.info(
            "Selected segment IDs: %s",
            sorted(selected_segment_ids) if selected_segment_ids else "none",
        )

        for seg_id in selected_segment_ids:
            mask[label_map == seg_id] = 1.0

        # Return as [1, H, W] float32 tensor (ComfyUI MASK convention)
        return torch.from_numpy(mask).unsqueeze(0)
