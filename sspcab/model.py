# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.nn as nn
from paddle.vision.models import resnet18
from sspcab.sspcab import SSPCAB
import paddle


class ProjectionNet(nn.Layer):
    def __init__(self, pretrained=True, head_layers=None, num_classes=2):
        super(ProjectionNet, self).__init__()
        # self.resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=pretrained)
        if head_layers is None:
            head_layers = [512, 512, 512, 512, 512, 512, 512, 512, 128]
        self.resnet18 = resnet18(pretrained=pretrained)

        # create MPL head as seen in the code in: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        # TODO: check if this is really the right architecture
        last_layer = 512
        sequential_layers = []

        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1D(num_neurons))
            sequential_layers.append(nn.ReLU())
            last_layer = num_neurons

        # the last layer without activation

        head = nn.Sequential(
            *sequential_layers
        )
        self.resnet18.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)
        self.mse_loss = paddle.nn.MSELoss()
        self.SSPCAB = SSPCAB(head_layers[0]//2)
        self.net = nn.Sequential(*list(self.resnet18.children()))

    def forward(self, x):
        embeds = self.net[:-3](x)
        output_sspcab = self.SSPCAB(embeds)
        cost_sspcab = self.mse_loss(output_sspcab, embeds)
        embeds = self.net[-1](self.net[-3:-1](output_sspcab).squeeze(-1).squeeze(-1))
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return embeds, logits, cost_sspcab

    def freeze_resnet(self):
        # freez full resnet18
        for param in self.resnet18.parameters():
            param.trainable = False

        # unfreeze head:
        for param in self.resnet18.fc.parameters():
            param.trainable = True

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.trainable = True
