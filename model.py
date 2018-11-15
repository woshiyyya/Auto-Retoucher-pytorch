import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet, model_urls
import torch.utils.model_zoo as model_zoo


class ResNetWrapper(ResNet):
    def __init__(self):
        super(ResNetWrapper, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        self.output_size = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # [N, 2048, 2, 2]
        x = x.view(x.size(0), -1)
        self.output_size = x.size(-1)
        return x


class BackgroundMatcher(nn.Module):
    def __init__(self):
        super(BackgroundMatcher, self).__init__()
        self.background_reader = ResNetWrapper()
        self.portrait_reader = ResNetWrapper()
        self.scene_reader = ResNetWrapper()
        logit_size = 3 * self.scene_reader.output_size
        self.linear = nn.Linear(logit_size, 1)

    def forward(self, xb, xp, xs):
        xb = self.background_reader(xb)
        xp = self.portrait_reader(xp)
        xs = self.scene_reader(xs)
        x = torch.cat([xb, xp, xs], dim=-1)

        return x
