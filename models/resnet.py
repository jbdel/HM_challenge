import torch.nn as nn
from torchvision import models

class Model_Resnet(nn.Module):
    def __init__(self, args):
        super(Model_Resnet, self).__init__()

        self.net = models.resnet152(pretrained=True)
        for param in self.parameters():
            param.requires_grad = False

        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features=in_features, out_features=1)

    def forward(self, img, text):
        return self.net(img)
