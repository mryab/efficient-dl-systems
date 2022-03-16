import math
import functools

import torch
from torch import nn
import torch.nn.functional as F


def get_active_fn(name):
    """Select activation function."""
    active_fn = {
        'nn.ReLU6': functools.partial(nn.ReLU6, inplace=True),
        'nn.ReLU': functools.partial(nn.ReLU, inplace=True),
        'nn.LeakyReLU': functools.partial(nn.LeakyReLU, inplace=True),
    }[name]
    return active_fn


class ConvBNReLU(nn.Sequential):
    """Convolution-BatchNormalization-ActivateFn."""
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 use_bias=True,
                 norm_layer=nn.InstanceNorm2d,
                 norm_kwargs=None,
                 active_fn=None):
        if norm_kwargs is None:
            norm_kwargs = {}
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      0,
                      groups=groups,
                      bias=use_bias), norm_layer(out_planes, **norm_kwargs),
            active_fn())


class InvertedResidualChannels(nn.Module):
    """MobiletNetV2 building block."""
    def __init__(self,
                 inp,
                 res_channels,
                 dw_channels,
                 channels_reduction_factor,
                 res_kernel_sizes,
                 dw_kernel_sizes,
                 padding_type='reflect',
                 use_bias=True,
                 norm_layer=nn.InstanceNorm2d,
                 norm_kwargs=None,
                 dropout_rate=0.0,
                 active_fn=None):
        super(InvertedResidualChannels, self).__init__()
        if isinstance(res_kernel_sizes, int):
            res_kernel_sizes = [res_kernel_sizes]
        if res_channels is not None:
            assert type(res_channels) == int or len(res_channels) == len(
                res_kernel_sizes)
        if type(dw_kernel_sizes) == int:
            dw_kernel_sizes = [dw_kernel_sizes]
        if dw_channels is not None:
            assert type(dw_channels) == int or len(dw_channels) == len(
                dw_kernel_sizes)

        self.input_dim = inp
        if res_channels is None:
            self.res_channels = [
                inp // channels_reduction_factor for _ in res_kernel_sizes
            ]
        elif type(res_channels) == int:
            self.res_channels = [
                res_channels // channels_reduction_factor
                for _ in res_kernel_sizes
            ]
        else:
            assert len(res_channels) == len(res_kernel_sizes)
            self.res_channels = [
                c // channels_reduction_factor for c in res_channels
            ]
        if dw_channels is None:
            self.dw_channels = [
                inp // channels_reduction_factor for _ in dw_kernel_sizes
            ]
        elif type(dw_channels) == int:
            self.dw_channels = [
                dw_channels // channels_reduction_factor
                for _ in dw_kernel_sizes
            ]
        else:
            assert len(dw_channels) == len(dw_kernel_sizes)
            self.dw_channels = [
                c // channels_reduction_factor for c in dw_channels
            ]
        self.res_kernel_sizes = res_kernel_sizes
        self.dw_kernel_sizes = dw_kernel_sizes
        self.padding_type = padding_type
        self.use_bias = use_bias
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        self.dropout_rate = dropout_rate
        self.active_fn = active_fn

        if self.padding_type == 'reflect':
            self.pad = nn.ReflectionPad2d
        elif self.padding_type == 'replicate':
            self.pad = nn.ReplicationPad2d
        elif self.padding_type == 'zero':
            self.pad = functools.partial(nn.ConstantPad2d, value=0.0)
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)

        self.res_ops, self.dw_ops, self.pw_bn = self._build()

    def _build(self):
        _norm_kwargs = self.norm_kwargs \
            if self.norm_kwargs is not None else {}

        res_ops = nn.ModuleList()
        for idx, (midp,
                  k) in enumerate(zip(self.res_channels,
                                      self.res_kernel_sizes)):
            if midp == 0:
                continue
            layers = []
            layers.append(self.pad((k - 1) // 2))
            layers.append(
                ConvBNReLU(self.input_dim,
                           midp,
                           kernel_size=k,
                           use_bias=self.use_bias,
                           norm_layer=self.norm_layer,
                           norm_kwargs=_norm_kwargs,
                           active_fn=self.active_fn))
            layers.append(nn.Dropout(self.dropout_rate))
            layers.append(self.pad((k - 1) // 2))
            layers.append(
                nn.Conv2d(midp, self.input_dim, k, 1, 0, bias=self.use_bias))
            res_ops.append(nn.Sequential(*layers))

        dw_ops = nn.ModuleList()
        for idx, (midp,
                  k) in enumerate(zip(self.dw_channels, self.dw_kernel_sizes)):
            if midp == 0:
                continue
            layers = []
            layers.append(
                ConvBNReLU(self.input_dim,
                           midp,
                           kernel_size=1,
                           use_bias=self.use_bias,
                           norm_layer=self.norm_layer,
                           norm_kwargs=_norm_kwargs,
                           active_fn=self.active_fn))
            layers.extend([
                self.pad((k - 1) // 2),
                ConvBNReLU(midp,
                           midp,
                           kernel_size=k,
                           groups=midp,
                           use_bias=self.use_bias,
                           norm_layer=self.norm_layer,
                           norm_kwargs=_norm_kwargs,
                           active_fn=self.active_fn),
                nn.Dropout(self.dropout_rate),
                nn.Conv2d(midp, self.input_dim, 1, 1, 0, bias=self.use_bias),
            ])
            dw_ops.append(nn.Sequential(*layers))
        pw_bn = self.norm_layer(self.input_dim, **_norm_kwargs)

        return res_ops, dw_ops, pw_bn

    def forward(self, x):
        if len(self.res_ops) == 0 and len(self.dw_ops) == 0:
            return x
        tmp = sum([op(x) for op in self.res_ops]) + sum(
            [op(x) for op in self.dw_ops])
        tmp = self.pw_bn(tmp)
        return x + tmp


class InceptionGenerator(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf,
                 channels,
                 channels_reduction_factor,
                 kernel_sizes,
                 padding_type='reflect',
                 norm_layer=nn.InstanceNorm2d,
                 norm_momentum=0.1,
                 norm_epsilon=1e-5,
                 dropout_rate=0,
                 active_fn='nn.ReLU',
                 n_blocks=9):
        assert (n_blocks >= 0)
        assert len(kernel_sizes) == len(
            set(kernel_sizes)), 'no duplicate in kernel sizes is allowed.'
        super(InceptionGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        norm_kwargs = {'momentum': norm_momentum, 'eps': norm_epsilon}
        active_fn = get_active_fn(active_fn)

        down_sampling = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            down_sampling += [
                nn.Conv2d(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2 ** n_downsampling

        n_blocks1 = n_blocks // 3
        n_blocks2 = n_blocks1
        n_blocks3 = n_blocks - n_blocks1 - n_blocks2

        features = []
        for i in range(n_blocks1):
            features += [
                InvertedResidualChannels(
                    ngf * mult,
                    res_channels=channels,
                    dw_channels=channels,
                    channels_reduction_factor=channels_reduction_factor,
                    res_kernel_sizes=kernel_sizes,
                    dw_kernel_sizes=kernel_sizes,
                    padding_type=padding_type,
                    use_bias=use_bias,
                    norm_layer=norm_layer,
                    norm_kwargs=norm_kwargs,
                    dropout_rate=dropout_rate,
                    active_fn=active_fn)
            ]

        for i in range(n_blocks2):
            features += [
                InvertedResidualChannels(
                    ngf * mult,
                    res_channels=channels,
                    dw_channels=channels,
                    channels_reduction_factor=channels_reduction_factor,
                    res_kernel_sizes=kernel_sizes,
                    dw_kernel_sizes=kernel_sizes,
                    padding_type=padding_type,
                    use_bias=use_bias,
                    norm_layer=norm_layer,
                    norm_kwargs=norm_kwargs,
                    dropout_rate=dropout_rate,
                    active_fn=active_fn)
            ]

        for i in range(n_blocks3):
            features += [
                InvertedResidualChannels(
                    ngf * mult,
                    res_channels=channels,
                    dw_channels=channels,
                    channels_reduction_factor=channels_reduction_factor,
                    res_kernel_sizes=kernel_sizes,
                    dw_kernel_sizes=kernel_sizes,
                    padding_type=padding_type,
                    use_bias=use_bias,
                    norm_layer=norm_layer,
                    norm_kwargs=norm_kwargs,
                    dropout_rate=dropout_rate,
                    active_fn=active_fn)
            ]

        up_sampling = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            up_sampling += [
                nn.ConvTranspose2d(ngf * mult,
                                   int(ngf * mult / 2),
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1,
                                   bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        up_sampling += [nn.ReflectionPad2d(3)]
        up_sampling += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        up_sampling += [nn.Tanh()]
        self.down_sampling = nn.Sequential(*down_sampling)
        self.features = nn.Sequential(*features)
        self.up_sampling = nn.Sequential(*up_sampling)

    def forward(self, input):
        """Standard forward"""
        res = self.down_sampling(input)
        res = self.features(res)
        res = self.up_sampling(res)
        return res


class Discriminator(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        n_layers: int = 3,
        use_affine: bool = True
    ):
        super(Discriminator, self).__init__()
        self._in_channels = in_channels
        self._hidden_channels = hidden_channels
        self._n_layers = n_layers
        self._use_affine = use_affine
        
        self._kernel_size = 4
        self._use_conv_bias = False
        self._padding = 1
        
        blocks = [
            nn.Sequential(
                nn.Conv2d(
                    self._in_channels, self._hidden_channels,
                    kernel_size=self._kernel_size, stride=2,
                    padding=self._padding
                ),
                nn.LeakyReLU(0.2, inplace=True)
            )
        ]
        
        expand_ratio = prev_expand_ratio = 1
        for i in range(1, self._n_layers + 1):
            prev_expand_ratio = expand_ratio
            expand_ratio = min(2 ** i, 8)
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        prev_expand_ratio * self._hidden_channels,
                        expand_ratio * self._hidden_channels,
                        kernel_size=self._kernel_size,
                        stride=2 if i != self._n_layers else 1,
                        padding=self._padding,
                        bias=self._use_conv_bias,
                    ),
                    nn.BatchNorm2d(
                        num_features=expand_ratio * self._hidden_channels,
                        affine=self._use_affine
                    ),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        blocks.append(
            nn.Sequential(
                nn.Conv2d(
                    expand_ratio * self._hidden_channels,
                    out_channels=1, kernel_size=self._kernel_size,
                    stride=1, padding=self._padding
                )
            )
        )
        self._blocks = nn.Sequential(*blocks)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._blocks(input)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 dilation=(1, 1),
                 residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation[1],
                               bias=False,
                               dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False,
                 out_middle=False,
                 pool_size=28,
                 arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        if arch == 'C':
            self.conv1 = nn.Conv2d(3,
                                   channels[0],
                                   kernel_size=7,
                                   stride=1,
                                   padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(BasicBlock,
                                           channels[0],
                                           layers[0],
                                           stride=1)
            self.layer2 = self._make_layer(BasicBlock,
                                           channels[1],
                                           layers[1],
                                           stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(3,
                          channels[0],
                          kernel_size=7,
                          stride=1,
                          padding=3,
                          bias=False), nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True))

            self.layer1 = self._make_conv_layers(channels[0],
                                                 layers[0],
                                                 stride=1)
            self.layer2 = self._make_conv_layers(channels[1],
                                                 layers[1],
                                                 stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block,
                                       channels[4],
                                       layers[4],
                                       dilation=2,
                                       new_level=False)
        self.layer6 = None if layers[5] == 0 else \
            self._make_layer(block, channels[5], layers[5], dilation=4,
                             new_level=False)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else \
                self._make_layer(BasicBlock, channels[6], layers[6], dilation=2,
                                 new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else \
                self._make_layer(BasicBlock, channels[7], layers[7], dilation=1,
                                 new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(channels[7], layers[7], dilation=1)

        if num_classes > 0:
            self.avgpool = nn.AvgPool2d(pool_size)
            self.fc = nn.Conv2d(self.out_dim,
                                num_classes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    new_level=True,
                    residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample,
                  dilation=(1, 1) if dilation == 1 else
                  (dilation // 2 if new_level else dilation, dilation),
                  residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      residual=residual,
                      dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes,
                          channels,
                          kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation,
                          bias=False,
                          dilation=dilation),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()

        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        x = self.layer4(x)
        y.append(x)

        x = self.layer5(x)
        y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)

        if self.out_map:
            x = self.fc(x)
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

        if self.out_middle:
            return x, y
        else:
            return x


def drn_d_105(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-105']))
    return model


class DRNSeg(nn.Module):
    def __init__(self,
                 model_name,
                 classes,
                 pretrained_model=None,
                 pretrained=True,
                 use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn_d_105(pretrained=pretrained, num_classes=1000)

        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes, kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes,
                                    classes,
                                    16,
                                    stride=8,
                                    padding=4,
                                    output_padding=0,
                                    groups=classes,
                                    bias=False)
            self.fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up
    
    @staticmethod
    def fill_up_weights(up):
        w = up.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x
