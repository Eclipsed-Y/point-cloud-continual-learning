from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from torch.nn import Dropout

import numpy as np
import torch.nn as nn
import torch
from torch.nn import Dropout
from collections import OrderedDict
from copy import deepcopy


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))

class AlexNet(nn.Module):
    def __init__(self, taskcla = None):
        super(AlexNet, self).__init__()
        self.act = OrderedDict()
        self.map = []
        self.ksize = []
        self.in_channel = []
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        # self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.bn1 = nn.BatchNorm2d(64)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        # self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        self.bn2 = nn.BatchNorm2d(128)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        # self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        self.bn3 = nn.BatchNorm2d(256)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256 * self.smid * self.smid)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * self.smid * self.smid, 2048, bias=False)
        # self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = nn.Linear(2048, 2048, bias=False)
        # self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])

        self.taskcla = taskcla
        self.last = nn.Linear(2048, 10, bias = False)


    def feature_proj(self, x, proj):
        # (batch, out, k, k)
        new_feature = (x.transpose(1, 3) @ proj).transpose(1, 3)
        return new_feature

    def fc_feature_proj(self, x, proj):
        # (batch,feature)
        return x @ proj
    def features(self, x):
        bsz = deepcopy(x.size(0))
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))
        x = x.view(bsz, -1)
        x = self.fc1(x)
        # x = self.drop2(self.relu(self.bn4(x)))
        x = self.drop2(self.relu(x))
        x = self.fc2(x)
        # x = self.drop2(self.relu(self.bn5(x)))
        x = self.drop2(self.relu(x))
        return x
    def logits(self, x):
        return self.last(x)
    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

class AlexNetCL(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=10):
        super(AlexNetCL, self).__init__()

        self.alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        self.alexnet.eval()
        
        # Freeze all layers
        for param in self.alexnet.parameters() :
            param.requires_grad = False

        # https://medium.com/@YeseulLee0311/pytorch-transfer-learning-alexnet-how-to-freeze-some-layers-26850fc4ac7e
        # model.classifier
        # Out[10]:
        # Sequential(
        # (0): Dropout(p=0.5, inplace=False)
        # (1): Linear(in_features=9216, out_features=4096, bias=True)
        # (2): ReLU(inplace=True)
        # (3): Dropout(p=0.5, inplace=False)
        # (4): Linear(in_features=4096, out_features=4096, bias=True)
        # (5): ReLU(inplace=True)
        # (6): Linear(in_features=4096, out_features=1000, bias=True)
        # )

        # # TOOO : Maybe start from the next layer -> less parameters to fit :)
        # self.linear = nn.Sequential(
        #     nn.Linear(in_features=9216, out_features=256, bias=True),
        #     nn.ReLU(inplace=True),
        #     Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=256, out_features=128, bias=True),
        #     nn.ReLU(inplace=True),
        #     # nn.Linear(in_features=256, out_features=out_dim, bias=True),
        # )

        # TOOO : Maybe start from the next layer -> less parameters to fit :)
        self.linear = nn.Sequential(
            nn.Linear(in_features=4096, out_features=hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=hidden_dim, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            # nn.Linear(in_features=256, out_features=out_dim, bias=True),
        )

        self.last = nn.Linear(128, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.alexnet.features(x)
        x = self.alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.alexnet.classifier[:4](x)
        x = self.linear(x)
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def get_AlexNet():
    return AlexNet()

if __name__ == '__main__':
    model = AlexNetCL()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {n_trainable} trainable parameters")
    #
    # # Download an example image from the pytorch website
    # import urllib
    #
    # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    # try:
    #     urllib.URLopener().retrieve(url, filename)
    # except:
    #     urllib.request.urlretrieve(url, filename)
    #
    # # sample execution (requires torchvision)
    # from PIL import Image
    # from torchvision import transforms
    #
    # input_image = Image.open(filename)
    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # input_tensor = preprocess(input_image)
    # input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    #
    # # move the input and model to GPU for speed if available
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')
    #     model.to('cuda')
    #
    # with torch.no_grad():
    #     output = model(input_batch)
    # # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    # # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    # print(torch.nn.functional.softmax(output[0], dim=0))