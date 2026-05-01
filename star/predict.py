import sys

import torch
from PIL import Image
from torchvision import transforms

from star.config import CLASS_NAMES, IMAGE_SIZE, MODEL_FILE
from star.train import SimpleCNN


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def prepare_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def predict_image(image):
    model, device = load_model()
    image_tensor = prepare_image(image).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())

    return {
        "label": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i].item())
            for i in range(len(CLASS_NAMES))
        },
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m star.predict path/to/image.jpg")
        return

    image_path = sys.argv[1]
    image = Image.open(image_path).convert("RGB")
    result = predict_image(image)

    print("Prediction:", result["label"])
    print("Confidence:", f"{result['confidence']:.4f}")
    print("Probabilities:", result["probabilities"])


if __name__ == "__main__":
    main()