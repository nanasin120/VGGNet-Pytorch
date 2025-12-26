import torch
import torch.nn as nn
import torch.nn.functional as F

vggnet_a_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vggnet_b_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vggnet_c_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, (1, 256), 'M', 512, 512, (1, 512), 'M', 512, 512, (1, 512), 'M']
vggnet_d_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
vggnet_e_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


def build_layer(cfg):
    layers = []
    in_channels = 3

    for x in cfg:
        if x == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif isinstance(x, tuple):
            k, c = x
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=c, kernel_size=k, padding=0))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            in_channels = c
        else:
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(x))
            layers.append(nn.ReLU(inplace=True))
            in_channels = x

    return nn.Sequential(*layers)

class VGGNet(nn.Module):
    def __init__(self, mode):
        super(VGGNet, self).__init__()

        configs = {
            'A': vggnet_a_cfg,
            'B': vggnet_b_cfg,
            'C': vggnet_c_cfg,
            'D': vggnet_d_cfg,
            'E': vggnet_e_cfg
        }

        self.feature_layer = build_layer(configs[mode])

        self.classifier_layer = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )
    
    def forward(self, x):
        x = self.feature_layer(x)
        x = torch.flatten(x, 1)
        x = self.classifier_layer(x)
        return x
    