import torch
import torch.nn as nn
import torchvision.models as models

# ---------------------------------------------------------
# IMAGE MODEL ARCHITECTURE
# ---------------------------------------------------------

class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, out_dim=64, variant="pool2"):
        super().__init__()

        if variant == "pool2":
            conv = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(2),
            )
        else:
            conv = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(2),
            )

        self.net = nn.Sequential(
            conv,
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResNet18Backbone(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.proj = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(True),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        return self.proj(self.backbone(x).flatten(1))


class AttentionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class ImageDeepfakeDetector(nn.Module):
    FEAT_DIM = 448

    def __init__(self):
        super().__init__()

        self.diffusion_branch = SmallCNN(1, 64)
        self.edge_branch = SmallCNN(1, 64, "convconvpool")
        self.semantic_branch = ResNet18Backbone(256)
        self.acquisition_branch = SmallCNN(1, 64)
        self.attention = AttentionGate(self.FEAT_DIM)

        self.classifier = nn.Sequential(
            nn.Linear(self.FEAT_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, d, e, s, a):
        f1 = self.diffusion_branch(d)
        f2 = self.edge_branch(e)
        f3 = self.semantic_branch(s)
        f4 = self.acquisition_branch(a)

        feats = torch.cat([f1, f2, f3, f4], dim=1)
        feats = self.attention(feats)

        return self.classifier(feats), feats


# ---------------------------------------------------------
# AUDIO MODEL ARCHITECTURE
# ---------------------------------------------------------

class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(64 * 16, 64, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(128, 1)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.shape
        x = x.view(B, W, C * H)

        x, _ = self.lstm(x)

        a = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        x = torch.sum(x * a.unsqueeze(-1), dim=1)

        return self.fc(x)
