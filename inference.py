import os

import torch
import torch.nn.functional as F
from torchvision import transforms

from dataset import extract_frames
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
    transforms.ToPILImage(),

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
_sequence_length = DEFAULT_SEQUENCE_LENGTH


def load_model():

    global _model
    global _sequence_length

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

        _sequence_length = checkpoint.get(
            "sequence_length",
            DEFAULT_SEQUENCE_LENGTH
        )

    else:

        state_dict = checkpoint

    model = DeepFakeDetector().to(device)

    model.load_state_dict(
        state_dict
    )

    model.eval()

    _model = model

    return _model


def predict_video(video_path):

    model = load_model()

    frames = extract_frames(
        video_path,
        sequence_length=_sequence_length
    )

    tensor_frames = torch.stack([
        transform(frame)
        for frame in frames
    ])

    tensor_frames = tensor_frames.unsqueeze(
        0
    )

    tensor_frames = tensor_frames.to(
        device
    )

    with torch.no_grad():

        outputs = model(
            tensor_frames
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
            "Usage: python inference.py <video_path>"
        )

        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):

        print(
            "Video file not found:",
            video_path
        )

        sys.exit(1)

    result = predict_video(
        video_path
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