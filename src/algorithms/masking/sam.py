import torch

from torch import Tensor
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

SAM_REDIM_RECOMMANDED_SIZE = 1024

class SAMSegmenter: 
    model: torch.nn.Module
    predictor: SamPredictor

    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str | None = None):
        if device is None: 
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
        self.predictor = SamPredictor(self.model)

    def process_mask(self, image_tensor: Tensor, point_coords: Tensor, point_labels: Tensor) -> Tensor:
        """
        Génère un masque directement à partir d'un tenseur PyTorch sur le GPU.
        
        :param image_tensor: Tenseur PyTorch de forme (H, W, 3), type float32, valeurs [0.0, 1.0], déjà sur le device (ex: 'cuda').
        :param point_coords: Liste de coordonnées [[x1, y1], [x2, y2]]
        :param point_labels: Liste de labels [1, 0]
        :return: Tenseur PyTorch du masque (H, W) en float32 de 0.0 à 1.0 sur le GPU
        """

        original_shape = image_tensor.shape[:2]
        image_tensor_255 = image_tensor * 255.0
        input_image_torch = image_tensor_255.permute(2, 0, 1).unsqueeze(0)
        transform = ResizeLongestSide(SAM_REDIM_RECOMMANDED_SIZE)
        input_image_torch = transform.apply_image_torch(input_image_torch)
        self.predictor.set_torch_image(input_image_torch, original_image_size=original_shape)
        pts = torch.tensor(point_coords, dtype=torch.float32, device=self.device).unsqueeze(0)
        lbls = torch.tensor(point_labels, dtype=torch.float32, device=self.device).unsqueeze(0)
        pts_transformed = transform.apply_coords_torch(pts, original_shape)
        masks, _, _ = self.predictor.predict_torch(
            point_coords=pts_transformed,
            point_labels=lbls,
            multimask_output=False,
            boxes=None
        )
        mask_float32 = masks[0, 0].to(torch.float32)
        
        return mask_float32
