'''ShuffleNetV2 in PyTorch.

See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c,cc, _, _ = x.size()#[1, 72, 4, 28, 28]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c,1, 1, 1)
        return x * y

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )

def channel_shuffle(x, groups):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, depth, height, width)
    #permute
    x = x.permute(0,2,1,3,4,5).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, depth, height, width)
    return x
    
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2

        if self.stride == 1:
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                #nn.ReLU(inplace=True),
                HardSwish(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                HardSwish(inplace=True),
                #nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv3d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm3d(inp),
                # pw-linear
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                #nn.ReLU(inplace=True),
                HardSwish(inplace=True),
            )
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                #nn.ReLU(inplace=True),
                HardSwish(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                #nn.ReLU(inplace=True),
                HardSwish(inplace=True),
            )
          
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.shape[1]//2), :, :, :]
            x2 = x[:, (x.shape[1]//2):, :, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif self.stride == 2:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=101, sample_size=224, width_mult=1., in_channels=3):
        super(ShuffleNetV2, self).__init__()
        assert sample_size % 16 == 0
        
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.25:
            self.stage_out_channels = [-1, 24,  32,  64, 128, 1024]
        elif width_mult == 0.5:
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))

        # building first layer
        input_channel = self.stage_out_channels[1] #24
        self.conv1 = conv_bn(in_channels, input_channel, stride=(1,2,2))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)): #3
            numrepeat = self.stage_repeats[idxstage] #4
            output_channel = self.stage_out_channels[idxstage+2] #2
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride))
                input_channel = output_channel
                
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1x1_bn(input_channel, self.stage_out_channels[-1])
    
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

	    # building classifier
        self.classifier = nn.Sequential(
                            nn.Dropout(0.2),
                            nn.Linear(self.stage_out_channels[-1], num_classes)
                            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.features(out)
        out = self.conv_last(out)
        print(" F.avg_pool3d",out.size()) #[1, 1024, 1, 7, 7]
        print(out.data.size()[-3:]) #torch.Size([1, 7, 7])
        #out = F.avg_pool3d(out, out.data.size()[-3:])
        out = self.avgpool(out)
        print("after",out.size()) # torch.Size([1, 1024, 1, 1, 1])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        # python shufflenetv2hs.py


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = ShuffleNetV2(**kwargs)
    return model

def create_MobileNet(num_classes = 101,width_mult=1):
    return get_model(num_classes=num_classes,width_mult=width_mult)

if __name__ == "__main__":
    model = get_model(num_classes=101, sample_size=224, width_mult=1.)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print(model)

    input_var = Variable(torch.randn(1, 3, 16, 224, 224))
    output = model(input_var)
    print(output.shape)

# python ShuffleNetV2.py



