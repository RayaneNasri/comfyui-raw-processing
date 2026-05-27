import os
import json
import logging
import torch
import folder_paths # type: ignore

SAM_MODELS_DIR = os.path.join(folder_paths.models_dir, "sams")
os.makedirs(SAM_MODELS_DIR, exist_ok=True)
folder_paths.add_model_folder_path("sams", SAM_MODELS_DIR)

class SAMInteractiveNode:
    def __init__(self):
        self.model = None
        self.current_model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("sams")
        if not models:
            models = ["Aucun modèle trouvé"]

        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (models,), 
                "click_data": ("STRING", {"default": "[]", "multiline": True}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate_mask"
    CATEGORY = "SAM/Masking"

    def load_model(self, model_name):
        """Gestion propre du chargement et des erreurs"""
        if model_name == "Aucun modèle trouvé":
            raise ValueError("Erreur : Aucun modèle SAM n'est installé dans le dossier 'models/sams'.")

        model_path = folder_paths.get_full_path("sams", model_name)
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Erreur : Le modèle '{model_name}' est introuvable à l'emplacement : {model_path}")

        if self.model is None or self.current_model_name != model_name:
            logging.info(f"[SAM Node] Chargement du modèle {model_name}...")
            try:
                self.current_model_name = model_name
                logging.info("[SAM Node] Modèle chargé avec succès.")
            except RuntimeError as e:
                logging.error(f"[SAM Node] Erreur critique lors du chargement: {str(e)}")
                raise RuntimeError(f"Le modèle {model_name} est invalide ou incompatible. Détails dans la console.")
            except Exception as e:
                raise Exception(f"Erreur inattendue lors de la lecture du fichier .pth: {str(e)}")

    def generate_mask(self, image, model_name, click_data):
        self.load_model(model_name)

        try:
            points = json.loads(click_data)
        except json.JSONDecodeError:
            logging.warning("[SAM Node] Données de clics invalides. Utilisation d'un masque vide.")
            points = []

        if not points:
            return (torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32),)

        dummy_mask = torch.ones((1, image.shape[1], image.shape[2]), dtype=torch.float32)

        return (dummy_mask,)
    
NODE_CLASS_MAPPINGS = {
    "SAMInteractiveNode": SAMInteractiveNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMInteractiveNode": "SAM Masking (GPU)"
}