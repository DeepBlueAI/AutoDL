import logging
import sys
from collections import OrderedDict

import torch
import torchvision.models as models
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck

import skeleton

formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)

model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}
class ResNet18(models.ResNet):
    Block = BasicBlock

    def __init__(self, in_channels, num_classes=10, **kwargs):
        Block = BasicBlock
        super(ResNet18, self).__init__(Block, [2, 2, 2, 2], num_classes=num_classes, **kwargs)    # resnet18

        if in_channels == 3:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
            )
        elif in_channels == 1:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                skeleton.nn.CopyChannels(3),
            )
        else:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(3),
            )

        self.last_channels = 512 * Block.expansion
        self.conv1d = torch.nn.Sequential(
            skeleton.nn.Split(OrderedDict([
                ('skip', torch.nn.Sequential(
                    # torch.nn.AvgPool1d(3, stride=2, padding=1)
                )),
                ('deep', torch.nn.Sequential(
                    # torch.nn.Conv1d(self.last_channels, self.last_channels // 4,
                    #                 kernel_size=1, stride=1, padding=0, bias=False),
                    # torch.nn.BatchNorm1d(self.last_channels // 4),
                    # torch.nn.ReLU(inplace=True),
                    # torch.nn.Conv1d(self.last_channels // 4, self.last_channels // 4,
                    #                 kernel_size=5, stride=1, padding=2, groups=self.last_channels // 4, bias=False),
                    # torch.nn.BatchNorm1d(self.last_channels // 4),
                    # torch.nn.ReLU(inplace=True),
                    # torch.nn.Conv1d(self.last_channels // 4, self.last_channels,
                    #                 kernel_size=1, stride=1, padding=0, bias=False),
                    # torch.nn.BatchNorm1d(self.last_channels),

                    torch.nn.Conv1d(self.last_channels, self.last_channels,
                                    kernel_size=5, stride=1, padding=2, bias=False),
                    torch.nn.BatchNorm1d(self.last_channels),
                    torch.nn.ReLU(inplace=True),
                ))
            ])),
            skeleton.nn.MergeSum(),

            # torch.nn.Conv1d(self.last_channels, self.last_channels,
            #                 kernel_size=5, stride=1, padding=2, bias=False),
            # torch.nn.BatchNorm1d(self.last_channels),
            # torch.nn.ReLU(inplace=True),

            torch.nn.AdaptiveAvgPool1d(1)
        )

        # self.self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # self.pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.fc = torch.nn.Linear(self.last_channels, num_classes, bias=False)
        self._half = False
        self._class_normalize = True
        self._is_video = False

    def set_video(self, is_video=True, times=False):
        self._is_video = is_video
        if is_video:
            self.conv1d_prev = torch.nn.Sequential(
                skeleton.nn.SplitTime(times),
                skeleton.nn.Permute(0, 2, 1, 3, 4),
            )
            self.conv1d_post = torch.nn.Sequential(
            )

    def is_video(self):
        return self._is_video

    def init(self, model_dir=None, gain=1.):
        self.model_dir = model_dir if model_dir is not None else self.model_dir
        sd = model_zoo.load_url(model_urls['resnet18'], model_dir=self.model_dir)
        # sd = model_zoo.load_url(model_urls['resnet34'], model_dir='./models/')
        del sd['fc.weight']
        del sd['fc.bias']
        self.load_state_dict(sd, strict=False)

        # for idx in range(len(self.stem)):
        #     m = self.stem[idx]
        #     if hasattr(m, 'weight') and not isinstance(m, torch.nn.BatchNorm2d):
        #         # torch.nn.init.kaiming_normal_(self.stem.weight, mode='fan_in', nonlinearity='linear')
        #         torch.nn.init.xavier_normal_(m.weight, gain=gain)
        #         LOGGER.debug('initialize stem weight')
        #
        # for idx in range(len(self.conv1d)):
        #     m = self.conv1d[idx]
        #     if hasattr(m, 'weight') and not isinstance(m, torch.nn.BatchNorm1d):
        #         # torch.nn.init.kaiming_normal_(self.stem.weight, mode='fan_in', nonlinearity='linear')
        #         torch.nn.init.xavier_normal_(m.weight, gain=gain)
        #         LOGGER.debug('initialize conv1d weight')

        # torch.nn.init.kaiming_uniform_(self.fc.weight, mode='fan_in', nonlinearity='sigmoid')
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        LOGGER.debug('initialize classifier weight')

    def forward_origin(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)

        if self.is_video():
            x = self.conv1d_prev(x)
            x = x.view(x.size(0), x.size(1), -1)
            x = self.conv1d(x)
            x = self.conv1d_post(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):  # pylint: disable=arguments-differ
        dims = len(inputs.shape)

        if self.is_video() and dims == 5:
            batch, times, channels, height, width = inputs.shape
            inputs = inputs.view(batch*times, channels, height, width)

        inputs = self.stem(inputs)
        logits = self.forward_origin(inputs)
        logits /= tau

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)):
            pos = (targets == 1).to(logits.dtype)
            neg = (targets < 1).to(logits.dtype)
            npos = pos.sum()
            nneg = neg.sum()

            positive_ratio = max(0.1, min(0.9, (npos) / (npos + nneg)))
            negative_ratio = max(0.1, min(0.9, (nneg) / (npos + nneg)))
            LOGGER.debug('[BCEWithLogitsLoss] positive_ratio:%f, negative_ratio:%f',
                         positive_ratio, negative_ratio)

            normalized_loss =  (loss * pos) / positive_ratio
            normalized_loss += (loss * neg) / negative_ratio

            loss = normalized_loss

        if reduction == 'avg':
            loss = loss.mean()
        elif reduction == 'max':
            loss = loss.max()
        elif reduction == 'min':
            loss = loss.min()
        return logits, loss

    def half(self):
        # super(BasicNet, self).half()
        for module in self.modules():
            if len([c for c in module.children()]) > 0:
                continue

            if not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.half()
            else:
                module.float()
        self._half = True
        return self

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            torch.nn.BatchNorm2d(out_planes),
            torch.nn.ReLU6(inplace=True)
        )


class InvertedResidual(torch.nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            torch.nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(oup),
        ])
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(torch.nn.Module):
    def __init__(self,
                 in_channels, num_classes=10,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        super(MobileNetV2, self).__init__()
        self._half = False
        self._class_normalize = True
        self._is_video = False
        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = torch.nn.Sequential(*features)

        # building classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.last_channel, num_classes),
        )
        if in_channels == 3:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
            )
        elif in_channels == 1:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                skeleton.nn.CopyChannels(3),
            )
        else:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(3),
            )
        self.conv1d = torch.nn.Sequential(
            skeleton.nn.Split(OrderedDict([
                ('skip', torch.nn.Sequential(
                )),
                ('deep', torch.nn.Sequential(
                    torch.nn.Conv1d(last_channel, last_channel,
                                    kernel_size=5, stride=1, padding=2, bias=False),
                    torch.nn.BatchNorm1d(last_channel),
                    torch.nn.ReLU(inplace=True),
                ))
            ])),
            skeleton.nn.MergeSum(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
    def set_video(self, is_video=True, times=False):
        self._is_video = is_video
        if is_video:
            self.conv1d_prev = torch.nn.Sequential(
                skeleton.nn.SplitTime(times),
                skeleton.nn.Permute(0, 2, 1, 3, 4),
            )
            self.conv1d_post = torch.nn.Sequential(
            )
    def is_video(self):
        return self._is_video

    def init(self, model_dir, gain=1.):
        sd = model_zoo.load_url(model_urls['mobilenet_v2'], model_dir=model_dir)

        del sd['classifier.1.weight']
        del sd['classifier.1.bias']
        self.load_state_dict(sd, strict=False)
        for idx in range(len(self.classifier)):
            m = self.classifier[idx]
            if hasattr(m, 'weight'):
                torch.nn.init.xavier_normal_(m.weight, gain=gain)
                LOGGER.debug('initialize classifier weight')

    def forward_origin(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = x.view(x.size(0), x.size(1), 1,1)
        if self.is_video():
            x = self.conv1d_prev(x)
            x = x.view(x.size(0), x.size(1), -1)
            x = self.conv1d(x)
            x = self.conv1d_post(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):  # pylint: disable=arguments-differ
        dims = len(inputs.shape)

        if self.is_video() and dims == 5:
            batch, times, channels, height, width = inputs.shape
            inputs = inputs.view(batch*times, channels, height, width)

        inputs = self.stem(inputs)
        logits = self.forward_origin(inputs)
        logits /= tau

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)):
            pos = (targets == 1).to(logits.dtype)
            neg = (targets < 1).to(logits.dtype)
            npos = pos.sum()
            nneg = neg.sum()

            positive_ratio = max(0.1, min(0.9, (npos) / (npos + nneg)))
            negative_ratio = max(0.1, min(0.9, (nneg) / (npos + nneg)))
            LOGGER.debug('[BCEWithLogitsLoss] positive_ratio:%f, negative_ratio:%f',
                         positive_ratio, negative_ratio)

            normalized_loss =  (loss * pos) / positive_ratio
            normalized_loss += (loss * neg) / negative_ratio

            loss = normalized_loss

        if reduction == 'avg':
            loss = loss.mean()
        elif reduction == 'max':
            loss = loss.max()
        elif reduction == 'min':
            loss = loss.min()
        return logits, loss

class VGG16(models.VGG):
    def __init__(self, in_channels, num_classes=10, **kwargs):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        Block = BasicBlock
        super(VGG16, self).__init__(self.make_layers(cfg, batch_norm=True), **kwargs)  # VGG16
        self.norm = None
        if in_channels == 3:
            self.stem = torch.nn.Sequential(
            )
        elif in_channels == 1:
            self.stem = torch.nn.Sequential(
                skeleton.nn.CopyChannels(3),
            )
        else:
            self.stem = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False)
            )

        # self.fc = torch.nn.Linear(512 * Block.expansion, num_classes, bias=False)
        self._half = False
        self._class_normalize = True
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_classes),
        )

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(512, 4096),
        #     torch.nn.ReLU(True),
        #     torch.nn.Dropout(),
        #     torch.nn.Linear(4096, 4096),
        #     torch.nn.ReLU(True),
        #     torch.nn.Dropout(),
        #     torch.nn.Linear(4096, num_classes),
        # )

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, torch.nn.ReLU(inplace=True)]
                in_channels = v
        return torch.nn.Sequential(*layers)

    def init(self, model_dir, gain=1.):
        sd = model_zoo.load_url(model_urls['vgg16_bn'], model_dir=model_dir)
        del sd['classifier.0.weight']
        del sd['classifier.0.bias']
        del sd['classifier.6.weight']
        del sd['classifier.6.bias']
        # del sd['fc.bias']
        self.load_state_dict(sd, strict=False)

        # for idx in range(len(self.stem)):
        #     m = self.stem[idx]
        #     if hasattr(m, 'weight'):
        #         torch.nn.init.xavier_normal_(m.weight, gain=gain)
        #         LOGGER.debug('initialize stem weight')
        for idx in range(len(self.classifier)):
            m = self.classifier[idx]
            if hasattr(m, 'weight'):
                torch.nn.init.xavier_normal_(m.weight, gain=gain)
        LOGGER.debug('initialize classifier weight')

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):  # pylint: disable=arguments-differ
        # inputs = self.norm(inputs)
        inputs = self.stem(inputs)
        inputs = self.features(inputs)
        inputs = self.avgpool(inputs)
        inputs = torch.flatten(inputs, 1)
        logits = self.classifier(inputs)
        # logits = models.VGG.forward(self, inputs)
        logits /= tau

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (
                torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)):
            pos = (targets == 1).to(logits.dtype)
            neg = (targets < 1).to(logits.dtype)
            npos = pos.sum()
            nneg = neg.sum()

            positive_ratio = max(0.1, min(0.9, (npos) / (npos + nneg)))
            negative_ratio = max(0.1, min(0.9, (nneg) / (npos + nneg)))
            LOGGER.debug('[BCEWithLogitsLoss] positive_ratio:%f, negative_ratio:%f',
                         positive_ratio, negative_ratio)

            normalized_loss = (loss * pos) / positive_ratio
            normalized_loss += (loss * neg) / negative_ratio

            loss = normalized_loss

        if reduction == 'avg':
            loss = loss.mean()
        elif reduction == 'max':
            loss = loss.max()
        elif reduction == 'min':
            loss = loss.min()
        return logits, loss

    def half(self):
        for module in self.modules():
            if len([c for c in module.children()]) > 0:
                continue

            if not isinstance(module, torch.nn.BatchNorm2d):
                module.half()
            else:
                module.float()
        self._half = True
        return self
