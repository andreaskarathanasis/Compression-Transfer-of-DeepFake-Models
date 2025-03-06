import torch
import torch.nn as nn

class Model_Single_Conv_Adapter_440k(nn.Module):
    def __init__(self, original_model=None):
        super(Model_Single_Conv_Adapter_440k, self).__init__()
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.layer5 = original_model.layer5
        self.convadapter = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.fc = original_model.fc
        self.fc1 = original_model.fc1

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.convadapter(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        return out
