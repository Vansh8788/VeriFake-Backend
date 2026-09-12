import os

import torch
import torch.nn.functional as F

from torchvision import transforms

from PIL import Image

from model import DeepFakeDetector


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "deepfake_detector.pt"
)


DEFAULT_SEQUENCE_LENGTH = 20


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


_model = None


def load_model():

    global _model

    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model = DeepFakeDetector().to(
        device
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    _model = model

    return _model


def predict_image(
    image_path,
    sequence_length=DEFAULT_SEQUENCE_LENGTH
):

    model = load_model()

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    )

    frames = image_tensor.unsqueeze(
        0
    ).repeat(
        sequence_length,
        1,
        1,
        1
    )

    frames = frames.unsqueeze(
        0
    )

    frames = frames.to(
        device
    )

    with torch.no_grad():

        outputs = model(
            frames
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )[0]

    real_probability = probabilities[
        0
    ].item()

    fake_probability = probabilities[
        1
    ].item()

    if fake_probability > real_probability:

        label = "FAKE"

        confidence = fake_probability

    else:

        label = "REAL"

        confidence = real_probability

    return {
        "label": label,
        "confidence": confidence,
        "real_probability": real_probability,
        "fake_probability": fake_probability
    }


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage: python image_inference.py <image_path>"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):

        print(
            "Image file not found:",
            image_path
        )

        sys.exit(1)

    result = predict_image(
        image_path
    )

    print()

    print(
        "Prediction:",
        result["label"]
    )

    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        "Real Probability:",
        f"{result['real_probability'] * 100:.2f}%"
    )

    print(
        "Fake Probability:",
        f"{result['fake_probability'] * 100:.2f}%"
    )