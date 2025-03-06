import torch
import torch.nn as nn

class Decomposed_4point5M(nn.Module):
    def __init__(self, original_model, num_classes=2, rank=6):
        super(Decomposed_4point5M, self).__init__()
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.layer5 = original_model.layer5
        self.layer6 = original_model.layer6
        self.layer7 = original_model.layer7
        self.layer8 = original_model.layer8
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*128, rank, bias=False),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(rank, 256, bias=True),
            nn.ReLU())
        self.fc2 = original_model.fc1
        self.fc3 = original_model.fc2

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
