import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet, model_urls
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


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


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Attention(nn.Module):
    def __init__(self, m, n):
        super(Attention, self).__init__()
        self.m = m
        self.n = n
        self.proj_1 = Parameter(torch.Tensor(30, m))
        self.proj_2 = Parameter(torch.Tensor(30, n))
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.proj_1.size(1))
        self.proj_1.data.uniform_(-stdv1, stdv1)
        stdv2 = 1. / math.sqrt(self.proj_2.size(1))
        self.proj_2.data.uniform_(-stdv2, stdv2)

    def forward(self, input1, input2, input3):
        N = input1.shape[0]
        h = torch.cat([input1, input2], dim=-1)
        proj1 = F.linear(h, self.proj_1).unsqueeze(-1)
        proj2 = F.linear(input3, self.proj_2).unsqueeze(-2)
        return gelu(torch.matmul(proj1, proj2).view(N, -1))


class Verifier(nn.Module):
    def __init__(self, config):
        super(Verifier, self).__init__()
        self.background_reader = ResNetWrapper()
        self.portrait_reader = ResNetWrapper()
        self.scene_reader = ResNetWrapper()
        self.config = config
        logit_size = 8192
        print("logits:", logit_size)
        self.maxpool = torch.nn.MaxPool1d(3)
        self.context_attn = Attention(2 * logit_size, logit_size)
        self.spatial_attn = Attention(2 * logit_size, logit_size)
        self.linear1 = nn.Linear(3 * logit_size + 900, 2)
        self.linear2 = nn.Linear(3 * logit_size + 900, 2)

    def forward(self, batch):
        xb = self.background_reader(batch['BGD'])
        xf = self.portrait_reader(batch['FGD'])
        xs = self.scene_reader(batch['SPS'])

        xb = F.dropout(xb, p=self.config['dropout'])
        xf = F.dropout(xf, p=self.config['dropout'])
        xs = F.dropout(xs, p=self.config['dropout'])

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
        x_cattn = self.context_attn(xb, xf, xn)
        x_sattn = self.spatial_attn(xs, xf, xn)
        print(x_sattn.shape)

        input_1 = torch.cat([xb, xf, xn, x_cattn], dim=-1)
        input_2 = torch.cat([xs, xf, xn, x_sattn], dim=-1)

        if self.config['Regression']:
            logit_1 = torch.sigmoid(self.linear1(input_1))
            logit_2 = torch.sigmoid(self.linear2(input_2))
        else:
            logit_1 = torch.softmax(self.linear1(input_1), dim=-1)
            logit_2 = torch.softmax(self.linear2(input_2), dim=-1)

        return logit_1, logit_2
