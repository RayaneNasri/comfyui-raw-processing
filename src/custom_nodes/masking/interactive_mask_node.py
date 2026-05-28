"""
node.py — Python backend for InteractiveSegmentationMask
=========================================================
Responsibilities
----------------
1. Receive an image tensor from the ComfyUI pipeline.
2. Run the requested segmentation engine (SLIC or SAM) to produce a
   label map: a 2-D integer array where every pixel carries its segment ID.
3. Build two auxiliary images and broadcast them to the frontend via
   ComfyUI's WebSocket server:
     a. **Overlay image** — the original image with segment boundaries
        drawn on top (for visual reference).
     b. **ID-map image**  — an RGB image where each segment is flood-filled
        with a unique, deterministic colour.  The JS reads this off-screen
        to resolve hover/click → segment-ID in O(1).
4. Receive the `selected_coords` JSON string that the JS widget writes
   after the user clicks segments.
5. Map every stored (x, y) coordinate back to a segment ID, union the
   matching pixels across all selected segments, and return a single
   binary MASK tensor shaped [1, H, W].

Architecture
------------
Each logical step lives in its own private method to keep concerns cleanly
separated and to give you obvious injection points for your real models.

WebSocket communication
-----------------------
Unlike HTTP polling, the backend broadcasts segment data via ComfyUI's
WebSocket server immediately after segmentation. The frontend listens on
the custom event channel "interactive_segmask" and updates the node display
in real-time, providing instant visual feedback.

Batch handling
--------------
ComfyUI images are [B, H, W, C] float32 tensors in the range [0, 1].
We always preview / segment only the FIRST frame (index 0) but we safely
handle the batch dimension throughout.
"""

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

# ComfyUI server import — guarded so the module can be imported in isolation
# (e.g. during unit tests) without a running ComfyUI instance.
try:
    from server import PromptServer # type: ignore

    _SERVER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SERVER_AVAILABLE = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Colour used to draw segment boundary lines on the overlay image.
_BOUNDARY_COLOUR: Tuple[int, int, int, int] = (255, 80, 0, 220)  # vivid orange

# Minimum contrast between consecutive ID-map colours (avoids neighbours
# looking identical at a glance — purely cosmetic, not relied on by logic).
_ID_MAP_SATURATION: int = 180

# How long (seconds) we cache segment data per node_id before evicting.
_CACHE_TTL: int = 600  # 10 minutes


# ---------------------------------------------------------------------------
# In-memory cache for label_maps only (to avoid re-segmenting identical images)
# ─────────────────────────────────────────────────────────────────────────
# Keyed by (node_id, image_hash) tuple. Each entry holds:
#   {
#     "label_map": np.ndarray  [H, W] int32,
#     "timestamp": float       time.time() of last write,
#   }

_label_map_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
_cache_lock = threading.Lock()


def _evict_stale_cache() -> None:
    """Remove entries older than _CACHE_TTL.  Called lazily on each write."""
    now = time.time()
    stale = [
        k for k, v in _label_map_cache.items()
        if now - v["timestamp"] > _CACHE_TTL
    ]
    for k in stale:
        del _label_map_cache[k]
        log.debug("Evicted stale label_map cache for key=%s", k)


# ---------------------------------------------------------------------------
# WebSocket event broadcasting
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Segmentation stubs
# ---------------------------------------------------------------------------
# Replace the body of each function with your real implementation.
# The contract (inputs / outputs) must remain identical.


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

    ─────────────────────────────────────────────────────────────────────────
    INJECT YOUR SLIC CODE HERE
    ─────────────────────────────────────────────────────────────────────────
    Example using scikit-image:

        from skimage.segmentation import slic
        label_map = slic(
            image_rgb,
            n_segments=kwargs.get("n_segments", 200),
            compactness=kwargs.get("compactness", 10),
            sigma=kwargs.get("sigma", 1),
            start_label=0,
        ).astype(np.int32)
        return label_map

    ─────────────────────────────────────────────────────────────────────────
    """
    log.info("Executing true SLIC segmentation via skimage.")
    
    # The placeholder was a 20x20 grid (400 segments).
    # compactness controls the balance between color proximity and space proximity.
    # 10.0 is a standard default, making superpixels relatively regular.
    label_map = slic(
        image_rgb, 
        n_segments=400, 
        compactness=5.0, 
        start_label=0,
        enforce_connectivity=True
    )
    
    # Ensure the output is a 32-bit integer array, which your ID-Map generator
    # (Claude's frontend logic) likely expects for color-coding.
    return label_map.astype(np.int32)


def _run_sam_segmentation(image_rgb: np.ndarray, **kwargs) -> np.ndarray:
    """
    Run Segment Anything Model (SAM) on *image_rgb* and return a label map.

    Parameters
    ----------
    image_rgb : np.ndarray
        uint8 array shaped [H, W, 3].
    **kwargs
        Additional hyper-parameters (e.g. points_per_side, pred_iou_thresh,
        sam_checkpoint path).

    Returns
    -------
    label_map : np.ndarray
        int32 array shaped [H, W].  Background pixels should be 0.

    ─────────────────────────────────────────────────────────────────────────
    INJECT YOUR SAM CODE HERE
    ─────────────────────────────────────────────────────────────────────────
    Example using the official SAM library:

        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry["vit_h"](checkpoint=kwargs["checkpoint"])
        sam.to(device=kwargs.get("device", "cuda"))
        generator = SamAutomaticMaskGenerator(sam)
        masks = generator.generate(image_rgb)
        # masks is a list of dicts; each dict has a boolean "segmentation" array.
        H, W = image_rgb.shape[:2]
        label_map = np.zeros((H, W), dtype=np.int32)
        for idx, mask_dict in enumerate(masks, start=1):
            label_map[mask_dict["segmentation"]] = idx
        return label_map

    ─────────────────────────────────────────────────────────────────────────
    """
    log.info("SAM segmentation stub called — returning placeholder radial segments.")
    H, W = image_rgb.shape[:2]
    # Placeholder: 16 radial "pie slice" segments centred on the image.
    cy, cx = H / 2, W / 2
    ys, xs = np.mgrid[0:H, 0:W]
    angles = np.arctan2(ys - cy, xs - cx)  # [-π, π]
    n_slices = 16
    label_map = ((angles + np.pi) / (2 * np.pi) * n_slices).astype(np.int32)
    return label_map


# ---------------------------------------------------------------------------
# Core node class
# ---------------------------------------------------------------------------


class InteractiveSegmentationMask:
    """
    ComfyUI node: Interactive Segmentation Selection Mask
    ======================================================
    Workflow
    --------
    1. The node receives an IMAGE tensor and widget parameters.
    2. It computes or retrieves (from cache) a segmentation label map.
    3. It generates overlay + ID-map images and broadcasts them to the
       frontend via ComfyUI's WebSocket server.
    4. On subsequent runs where the user has already clicked segments,
       `selected_coords` carries a JSON list of {x, y} dicts.  Each
       coordinate is mapped to a segment ID via the label map, and
       all matching pixels are OR-ed together into a single binary mask.
    5. The mask is returned as a [1, H, W] float32 tensor (ComfyUI MASK).

    WebSocket communication
    ----------------------
    After segmentation completes, the backend broadcasts the generated images
    via `PromptServer.send_sync()` on the "interactive_segmask" channel.
    The frontend listens for these events and updates the node preview
    automatically, enabling real-time visual feedback without HTTP polling.

    Widget layout visible in the ComfyUI UI
    ----------------------------------------
    • image             — incoming IMAGE connection
    • segmentation_engine — combo: SLIC | SAM
    • selected_coords   — hidden STRING, managed by JS (not shown to user)
    """

    # ------------------------------------------------------------------
    # ComfyUI class-level metadata
    # ------------------------------------------------------------------

    CATEGORY = "masking/interactive"
    FUNCTION = "execute"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    # This node is NOT pure (its output depends on hidden JS-managed state)
    # so ComfyUI must always re-execute it even when visible inputs are unchanged.
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
                # JSON string written by JS, e.g.:
                # '[{"x": 120, "y": 45}, {"x": 300, "y": 200}]'
                # An empty string or "[]" means nothing is selected.
                "selected_coords": ("STRING", {"default": "[]"}),
            },
            "hidden": {
                # Unique node identifier injected by ComfyUI automatically.
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
        selected_coords = kwargs.get("selected_coords", "")
        return selected_coords

    # ------------------------------------------------------------------
    # Main entry-point called by the ComfyUI execution engine
    # ------------------------------------------------------------------

    def execute(
        self,
        image: torch.Tensor,
        segmentation_engine: str,
        selected_coords: str = "[]",
        unique_id: str = "0",
    ) -> Tuple[torch.Tensor]:
        """
        Execute the node.

        Parameters
        ----------
        image : torch.Tensor
            Shape [B, H, W, C], dtype float32, range [0, 1].
        segmentation_engine : str
            "SLIC" or "SAM".
        selected_coords : str
            JSON array of {x, y} objects representing user-clicked pixels.
        unique_id : str
            ComfyUI node UID — used as cache key.

        Returns
        -------
        Tuple containing a single MASK tensor of shape [1, H, W], float32,
        values in {0.0, 1.0}.
        """
        log.info(
            "InteractiveSegmentationMask.execute: engine=%s, node_id=%s",
            segmentation_engine,
            unique_id,
        )

        # ── 1. Convert tensor to uint8 numpy image ─────────────────────
        image_np = _tensor_to_numpy_uint8(image)  # [H, W, C]
        H, W = image_np.shape[:2]
        image_hash = _image_tensor_hash(image)

        # ── 2. Segmentation (cached if image unchanged) ─────────────────
        label_map = self._get_or_compute_segments(
            image_np=image_np,
            engine=segmentation_engine,
            node_id=unique_id,
            image_hash=image_hash,
        )

        # ── 3. Build and cache frontend data ────────────────────────────
        self._update_frontend_cache(
            node_id=unique_id,
            image_np=image_np,
            label_map=label_map,
            image_hash=image_hash,
        )

        # ── 4. Build mask from user selections ──────────────────────────
        mask_tensor = self._build_mask_from_selections(
            label_map=label_map,
            selected_coords_json=selected_coords,
            H=H,
            W=W,
        )

        return (mask_tensor,)

    # ------------------------------------------------------------------
    # Step 2 — Segmentation
    # ------------------------------------------------------------------

    def _get_or_compute_segments(
        self,
        image_np: np.ndarray,
        engine: str,
        node_id: str,
        image_hash: int,
    ) -> np.ndarray:
        """
        Return a cached label map if the image hasn't changed; otherwise
        re-run the chosen segmentation engine.

        Parameters
        ----------
        image_np : np.ndarray  [H, W, 3] uint8
        engine   : str         "SLIC" | "SAM"
        node_id  : str         Node identifier
        image_hash : int       Cheap hash of the source tensor

        Returns
        -------
        label_map : np.ndarray [H, W] int32
        """
        cache_key = (node_id, image_hash)
        with _cache_lock:
            cached = _label_map_cache.get(cache_key)
            if cached is not None:
                log.debug("Label_map cache hit for node_id=%s", node_id)
                return cached["label_map"]

        log.info("Running %s segmentation for node_id=%s …", engine, node_id)

        # Only RGB is passed to the segmentation engines.
        image_rgb = image_np[..., :3] if image_np.shape[2] >= 3 else image_np

        if engine == "SLIC":
            label_map = _run_slic_segmentation(image_rgb)
        elif engine == "SAM":
            label_map = _run_sam_segmentation(image_rgb)
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

    # ------------------------------------------------------------------
    # Step 3 — Frontend image generation & broadcasting via WebSocket
    # ------------------------------------------------------------------

    def _update_frontend_cache(
        self,
        node_id: str,
        image_np: np.ndarray,
        label_map: np.ndarray,
        image_hash: int,
    ) -> None:
        """
        Generate the overlay and ID-map images, base-64 encode them, and
        broadcast them to the frontend via WebSocket.

        Both images share the same width/height as the source image.

        Overlay image
        -------------
        The original image with segment boundaries drawn as coloured lines.
        Computed with a simple erosion-based boundary detector (no scipy
        dependency).

        ID-map image
        ------------
        A solid-colour fill for every segment using `_deterministic_colour`.
        Each pixel's colour encodes the segment ID — the JS reads this
        off-screen to resolve hover/click events without any computation.
        """

        H, W = label_map.shape
        MAX_PREVIEW_SIZE = 512
        
        # Calcul du ratio de redimensionnement
        scale = min(MAX_PREVIEW_SIZE / W, MAX_PREVIEW_SIZE / H, 1.0)
        new_W, new_H = int(W * scale), int(H * scale)

        # ── Generate overlay ──────────────────────────────────────────────
        overlay_img = self._draw_boundary_overlay(image_np, label_map)
        if scale < 1.0:
            overlay_img = overlay_img.resize((new_W, new_H), Image.Resampling.LANCZOS)
        overlay_b64 = _pil_to_base64_png(overlay_img)

        # ── Generate ID map ──────────────────────────────────────────────
        id_map_img = self._draw_id_map(label_map)
        if scale < 1.0:
            # CRITIQUE : NEAREST pour ne pas altérer les couleurs uniques des segments
            id_map_img = id_map_img.resize((new_W, new_H), Image.Resampling.NEAREST)
        id_map_b64 = _pil_to_base64_png(id_map_img)

        # ── Broadcast via WebSocket ──────────────────────────────────────
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

    # ------------------------------------------------------------------
    # Step 4 — Mask generation from user selections
    # ------------------------------------------------------------------

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

        # ── Parse JSON safely ────────────────────────────────────────────
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

        # ── Resolve coordinates → segment IDs ────────────────────────────
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

        # ── Build union mask ─────────────────────────────────────────────
        for seg_id in selected_segment_ids:
            mask[label_map == seg_id] = 1.0

        # Return as [1, H, W] float32 tensor (ComfyUI MASK convention)
        return torch.from_numpy(mask).unsqueeze(0)