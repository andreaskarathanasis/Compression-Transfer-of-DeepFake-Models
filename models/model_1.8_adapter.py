import torch
import torch.nn as nn

class Model_Single_Conv_Adapter_1point8m(nn.Module):
    def __init__(self, original_model=None):
        super(Model_Single_Conv_Adapter_1point8m, self).__init__()
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.layer5 = original_model.layer5
        self.layer6 = original_model.layer6
        self.layer7 = original_model.layer7
        self.convadapter = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.fc = original_model.fc
        self.fc1 = original_model.fc1
        self.fc2 = original_model.fc2

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.convadapter(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
