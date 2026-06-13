import os
import logging
from PIL import Image
import torch
import timm
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")
MODEL_PATH = os.path.join(BASE_DIR, "cats_dogs_triplet_rexnet_150_best_model.pth")
logger = logging.getLogger(__name__)


class ModelException(Exception):
    """general exception for model errors."""


class CatDogBreedClassifier:
    """class which loads and predicts the breed of cats and dogs based on a pre-trained model."""
    def __init__(
        self,
        model_path="cats_dogs_triplet_rexnet_150_best_model.pth",
        class_names_path=None,
    ):
        """constror which initializes the model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = self._load_class_names(class_names_path)
        logger.info("Loaded %s class names.", len(self.class_names))

        self.model = timm.create_model("rexnet_150", num_classes=len(self.class_names))
        logger.info("Created model.")

        if not os.path.exists(model_path):
            raise ModelException(f"Model file not found at {model_path}.")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Loaded model from %s.", model_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_class_names(self, path):
        """load class names from file."""
        if path is None:
            path = CLASS_NAMES_PATH

        logger.info("Loading class names from %s.", path)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines()]

        raise ModelException(f"Class names file not found at {path}.")

    def predict(self, image_stream):
        """Predict breed from a file-like binary stream."""
        image_stream.seek(0)
        image = Image.open(image_stream).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        logger.info("Running inference.")
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=5, dim=1)

        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()

        results = []
        for idx, prob in zip(top_indices, top_probs):
            results.append({
                "class_idx": int(idx),
                "class_name": self.class_names[idx],
                "probability": float(prob),
            })

        return results
