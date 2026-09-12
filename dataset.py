import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def extract_frames(video_path, sequence_length=20):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Could not read video: {video_path}")

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        sequence_length,
        dtype=int
    )

    idx_set = set(frame_indices.tolist())

    frames = []
    current_frame = 0

    while cap.isOpened() and len(frames) < sequence_length:
        ret, frame = cap.read()

        if not ret:
            break

        if current_frame in idx_set:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame = cv2.resize(
                frame,
                (224, 224)
            )

            frames.append(frame)

        current_frame += 1

    cap.release()

    while len(frames) < sequence_length:
        if frames:
            frames.append(frames[-1])
        else:
            frames.append(
                np.zeros(
                    (224, 224, 3),
                    dtype=np.uint8
                )
            )

    return np.array(frames)


class VideoDataset(Dataset):

    def __init__(self, samples, sequence_length=20):
        self.samples = samples
        self.sequence_length = sequence_length

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        frames = extract_frames(
            video_path,
        self.sequence_length
        )

        tensor_frames = torch.stack([
        self.transform(frame)
        for frame in frames
        ])

        return tensor_frames, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    print("VideoDataset imported successfully.")
