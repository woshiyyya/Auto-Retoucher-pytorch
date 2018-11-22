import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet, model_urls
import torch.utils.model_zoo as model_zoo


class ResNetWrapper(ResNet):
    def __init__(self):
        super(ResNetWrapper, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        self.output_size = 2048

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class BackgroundMatcher(nn.Module):
    def __init__(self):
        super(BackgroundMatcher, self).__init__()
        self.background_reader = ResNetWrapper()
        self.portrait_reader = ResNetWrapper()
        self.scene_reader = ResNetWrapper()
        logit_size = self.scene_reader.output_size
        print("logits:", logit_size)
        self.maxpool = torch.nn.MaxPool1d(3)
        self.linear1 = nn.Linear(3 * logit_size, 2)
        self.linear2 = nn.Linear(3 * logit_size, 2)

    def forward(self, batch):
        xb = self.background_reader(batch['BGD'])
        xf = self.portrait_reader(batch['FGD'])
        xs = self.scene_reader(batch['SPS'])

        xbn = torch.unsqueeze(xb, dim=-2).transpose(-1, -2)
        xfn = torch.unsqueeze(xf, dim=-2).transpose(-1, -2)
        xsn = torch.unsqueeze(xs, dim=-2).transpose(-1, -2)
        # print("xb", xb.shape)
        # print("xf", xf.shape)
        # print("xs", xs.shape)
        # print("xbn", xbn.shape)
        # print("xfn", xfn.shape)
        # print("xsn", xsn.shape)

        xn = self.maxpool(torch.cat([xbn, xfn, xsn], dim=-1)).squeeze(-1)

        # print("xn", xn.shape)
        input_1 = torch.cat([xb, xf, xn], dim=-1)
        logit_1 = torch.softmax(self.linear1(input_1), dim=-1)

        input_2 = torch.cat([xs, xf, xn], dim=-1)
        logit_2 = torch.softmax(self.linear2(input_2), dim=-1)
        return logit_1, logit_2
