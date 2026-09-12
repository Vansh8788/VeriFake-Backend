import torch
import torch.nn as nn
from torchvision import models


class DeepFakeDetector(nn.Module):

    def __init__(self):
        super().__init__()

        resnext = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.DEFAULT
        )

        self.feature_extractor = nn.Sequential(
            *list(resnext.children())[:-1]
        )

        self.feature_dim = resnext.fc.in_features

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(512, 2)

    def extract_features(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        return features

    def forward(self, x):

        batch_size, seq_len, c, h, w = x.shape

        x = x.view(
            batch_size * seq_len,
            c,
            h,
            w
        )

        features = self.extract_features(x)

        features = features.view(
            batch_size,
            seq_len,
            self.feature_dim
        )

        lstm_out, (hidden, cell) = self.lstm(features)

        last_output = lstm_out[:, -1, :]

        out = self.dropout(last_output)

        out = self.classifier(out)

        return out


if __name__ == "__main__":

    model = DeepFakeDetector()

    print("Model created successfully.")
    print("Feature dimension:", model.feature_dim)

    x = torch.randn(
        2,
        20,
        3,
        224,
        224
    )

    print("Input shape:", x.shape)

    with torch.no_grad():
        output = model(x)

    print("Output shape:", output.shape)
    print("Output:", output)

    probabilities = torch.softmax(output, dim=1)

    print("Probabilities:", probabilities)

    predictions = torch.argmax(probabilities, dim=1)

    print("Predictions:", predictions)

    for i, prediction in enumerate(predictions):

        if prediction.item() == 1:
            result = "FAKE"
        else:
            result = "REAL"

        print(
            f"Video {i + 1}: {result}"
        )