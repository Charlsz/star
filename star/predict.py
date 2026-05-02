from pathlib import Path
import argparse

import torch
from torch import nn
from torchvision import transforms
from PIL import Image

from star.config import MODEL_FILE, IMAGE_SIZE


CLASS_NAMES = ["smooth", "featured"]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


def load_model(device):
    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    return model


def predict_image(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    transform = build_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = probabilities.argmax().item()

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[predicted_index].item()

    return predicted_class, confidence, probabilities.cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to the image")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    predicted_class, confidence, probabilities = predict_image(image_path, model, device)

    print(f"image={image_path}")
    print(f"predicted_class={predicted_class}")
    print(f"confidence={confidence:.4f}")
    print(f"smooth_probability={probabilities[0]:.4f}")
    print(f"featured_probability={probabilities[1]:.4f}")


if __name__ == "__main__":
    main()