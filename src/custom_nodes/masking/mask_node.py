import torch
import numpy as np
from PIL import Image
import base64
import io


class MaskDrawNode:
    """
    Nœud ComfyUI permettant de dessiner un masque directement dans l'interface.
    Le masque est dessiné via un widget canvas JS et transmis en base64.
    Retourne l'image originale + le masque comme tensor PyTorch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                # Ce widget est contrôlé côté JS — sa valeur est une string base64 PNG
                "mask_data": ("STRING", {"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "apply_mask"
    CATEGORY = "image/masking"
    OUTPUT_NODE = False

    def apply_mask(self, image: torch.Tensor, mask_data: str = "", unique_id=None):
        """
        image   : tensor (B, H, W, 3) float32 [0-1]
        mask_data: string base64 d'un PNG niveaux de gris (blanc = sélectionné)
        
        Retourne :
          image : tensor original inchangé (B, H, W, 3)
          mask  : tensor (B, H, W) float32 [0-1]
        """
        B, H, W, C = image.shape

        if not mask_data or mask_data.strip() == "":
            # Pas de masque dessiné → masque entièrement blanc (tout sélectionné)
            mask = torch.ones((B, H, W), dtype=torch.float32)
            return (image, mask)

        # Décoder le base64
        try:
            # Enlever le préfixe data URI si présent
            if "," in mask_data:
                mask_data = mask_data.split(",", 1)[1]

            raw = base64.b64decode(mask_data)
            pil_mask = Image.open(io.BytesIO(raw)).convert("L")  # Niveaux de gris

            # Redimensionner au format de l'image d'entrée si nécessaire
            if pil_mask.size != (W, H):
                pil_mask = pil_mask.resize((W, H), Image.LANCZOS)

            # Convertir en tensor float32 [0-1]
            mask_np = np.array(pil_mask, dtype=np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # (1, H, W)

            # Répliquer pour le batch
            mask_tensor = mask_tensor.expand(B, -1, -1)

        except Exception as e:
            print(f"[MaskDrawNode] Erreur décodage masque : {e}")
            mask_tensor = torch.ones((B, H, W), dtype=torch.float32)

        return (image, mask_tensor)

    @classmethod
    def IS_CHANGED(cls, image, mask_data="", unique_id=None):
        # Forcer la réévaluation à chaque changement de mask_data
        import hashlib
        return hashlib.md5(mask_data.encode()).hexdigest()


# ── Nœud utilitaire : applique un masque sur une image (composite) ──────────

class ApplyMaskToImage:
    """
    Applique un masque sur une image.
    Utile pour composer l'image masquée avant de la passer à un autre nœud.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "background_color": (["black", "white", "transparent"], {"default": "black"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("masked_image",)
    FUNCTION = "apply"
    CATEGORY = "image/masking"

    def apply(self, image: torch.Tensor, mask: torch.Tensor, background_color: str = "black"):
        """
        image : (B, H, W, 3)
        mask  : (B, H, W) ou (H, W)
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Aligner les dimensions de batch
        if mask.shape[0] != image.shape[0]:
            mask = mask.expand(image.shape[0], -1, -1)

        # Étendre le masque à 3 canaux
        mask_3ch = mask.unsqueeze(-1).expand_as(image)

        if background_color == "white":
            bg = torch.ones_like(image)
        else:
            bg = torch.zeros_like(image)

        result = image * mask_3ch + bg * (1.0 - mask_3ch)
        return (result,)


# ── Nœud utilitaire : inverse un masque ─────────────────────────────────────

class InvertMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",)}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("inverted_mask",)
    FUNCTION = "invert"
    CATEGORY = "image/masking"

    def invert(self, mask: torch.Tensor):
        return (1.0 - mask,)


# ── Enregistrement des nœuds ─────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "MaskDrawNode": MaskDrawNode,
    "ApplyMaskToImage": ApplyMaskToImage,
    "InvertMask": InvertMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskDrawNode": "Mask Draw",
    "ApplyMaskToImage": "Apply Mask to Image",
    "InvertMask": "Invert Mask",
}