import struct

import numpy as np
import torch
import torch.nn.functional as F

_OPCODE_GAIN_MAP = 9
# DNG OpcodeList tags (IFD tag IDs)
_DNG_TAG_OPCODE_LIST1 = 51020
_DNG_TAG_OPCODE_LIST2 = 51021


def try_read_vignette_gain_map(
    image_path: str | None,
    target_h: int,
    target_w: int,
) -> torch.Tensor | None:
    """
    Attempt to extract a vignetting gain map from DNG metadata.

    Looks for a GainMap opcode (ID 9) in the DNG tags. The raw map is bilinearly upsampled to (target_h, target_w, 3). Returns None silently if the path is empty, the file has no GainMap, or any parsing error occurs; the caller then falls back to manual parameters.

    Args:
        image_path (str | None): Path to the RAW/DNG file. Empty string or None returns None.
        target_h (int): Target height to upsample the gain map to.
        target_w (int): Target width to upsample the gain map to.

    Returns:
        torch.Tensor | None: Multiplicative gain map of shape (H, W, 3), or None if extraction fails.
    """
    if not image_path:
        return None

    try:
        import tifffile

        with tifffile.TiffFile(image_path) as tif:
            tags = tif.pages[0].tags  # type: ignore
            raw_data: bytes | None = None
            for tag_id in (_DNG_TAG_OPCODE_LIST1, _DNG_TAG_OPCODE_LIST2):
                tag = tags.get(tag_id)
                if tag is not None:
                    value = tag.value
                    raw_data = value if isinstance(value, bytes) else bytes(value)
                    break

            if raw_data is None:
                return None

            gain_maps = _parse_opcode_list(raw_data)
            if not gain_maps:
                return None

            return _assemble_gain_map(gain_maps, target_h, target_w)

    except Exception:
        return None


def _parse_opcode_list(data: bytes) -> list[np.ndarray]:
    """
    Extract all GainMap arrays from a DNG opcode list binary blob.

    Args:
        data (bytes): Binary opcode list data from DNG metadata tags.

    Returns:
        list[np.ndarray]: List of gain map arrays extracted from the opcode list. Returns empty list if parsing fails.
    """
    try:
        offset = 0
        (num_opcodes,) = struct.unpack_from(">I", data, offset)
        offset += 4
        gain_maps: list[np.ndarray] = []

        for _ in range(num_opcodes):
            if offset + 16 > len(data):
                break
            (opcode_id,) = struct.unpack_from(">I", data, offset)
            (param_len,) = struct.unpack_from(">I", data, offset + 12)
            params = data[offset + 16 : offset + 16 + param_len]
            offset += 16 + param_len

            if opcode_id == _OPCODE_GAIN_MAP:
                gm = _parse_gain_map_params(params)
                if gm is not None:
                    gain_maps.append(gm)

        return gain_maps
    except Exception:
        return []


def _parse_gain_map_params(params: bytes) -> np.ndarray | None:
    """
    Parse the binary parameters of a DNG GainMap opcode.

    Layout (big-endian):
        8×4 bytes  doubles: Top, Left, Bottom, Right
        4×4 bytes  uint32:  Plane, Planes, RowPitch, ColPitch
        2×4 bytes  uint32:  MapPointsV, MapPointsH
        4×8 bytes  doubles: MapSpacingV, MapSpacingH, MapOriginV, MapOriginH
        1×4 bytes  uint32:  MapPlanes
        N×4 bytes  float32: MapGain  (N = MapPointsV × MapPointsH × MapPlanes)
    """
    try:
        off = 32  # skip Top, Left, Bottom, Right (4 doubles)
        plane, planes, row_pitch, col_pitch = struct.unpack_from(">4I", params, off)
        off += 16
        map_pts_v, map_pts_h = struct.unpack_from(">2I", params, off)
        off += 8
        off += 32  # skip MapSpacingV/H, MapOriginV/H (4 doubles)
        (map_planes,) = struct.unpack_from(">I", params, off)
        off += 4

        n_values = map_pts_v * map_pts_h * map_planes
        gains = struct.unpack_from(f">{n_values}f", params, off)
        return np.array(gains, dtype=np.float32).reshape(
            map_pts_v, map_pts_h, map_planes
        )
    except Exception:
        return None


def _assemble_gain_map(
    gain_maps: list[np.ndarray],
    target_h: int,
    target_w: int,
) -> torch.Tensor | None:
    """
    Convert raw gain map array(s) to a target-sized RGB tensor.

    Handles RGGB Bayer (averaging G planes), 3-channel, and 1-channel gain maps. Upsamples to the target dimensions using bilinear interpolation.

    Args:
        gain_maps (list[np.ndarray]): List of gain map arrays extracted from DNG metadata.
        target_h (int): Target height for the upsampled gain map.
        target_w (int): Target width for the upsampled gain map.

    Returns:
        torch.Tensor | None: RGB gain map tensor of shape (target_h, target_w, 3), or None if conversion fails.
    """
    try:
        gm = gain_maps[0]  # (pts_v, pts_h, planes)

        if gm.shape[2] == 4:
            # RGGB Bayer: average the two G planes
            r = gm[..., 0:1]
            g = (gm[..., 1:2] + gm[..., 2:3]) / 2.0
            b = gm[..., 3:4]
            gm_rgb = np.concatenate([r, g, b], axis=-1)
        elif gm.shape[2] == 3:
            gm_rgb = gm
        elif gm.shape[2] == 1:
            gm_rgb = np.repeat(gm, 3, axis=-1)
        else:
            return None

        # (pts_v, pts_h, 3) → (1, 3, pts_v, pts_h) for interpolate
        t = torch.from_numpy(gm_rgb).permute(2, 0, 1).unsqueeze(0)
        upsampled = F.interpolate(
            t, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return upsampled.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
    except Exception:
        return None
