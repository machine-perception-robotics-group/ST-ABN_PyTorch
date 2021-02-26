import logging

import torch.nn as nn
import torch.utils.checkpoint as cp
import torch

from ....utils.misc import rgetattr, rhasattr
from .resnet import ResNet
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from ..utils.nonlocal_block import build_nonlocal_block
from ..spatial_temporal_modules.non_local import NonLocalModule

from ...registry import BACKBONES


def conv3x3x3(inplanes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(
        inplanes,
        out_planes,
        kernel_size=3,
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=dilation,
        dilation=dilation,
        bias=False)


def conv1x3x3(inplanes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "1x3x3 convolution with padding"
    return nn.Conv3d(
        inplanes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=(0, dilation, dilation),
        dilation=dilation,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 if_inflate=True,
                 inflate_style='3x1x1',
                 if_nonlocal=True,
                 nonlocal_cfg=None,
                 with_cp=True):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert inflate_style in ['3x1x1', '3x3x3']
        self.inplanes = inplanes
        self.planes = planes

        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride
        else:
            self.conv1_stride = spatial_stride
            self.conv2_stride = 1
            self.conv1_stride_t = temporal_stride
            self.conv2_stride_t = 1
        if if_inflate:
            if inflate_style == '3x1x1':
                self.conv1 = nn.Conv3d(
                    inplanes,
                    planes,
                    kernel_size=(3, 1, 1),
                    stride=(self.conv1_stride_t, self.conv1_stride, self.conv1_stride),
                    padding=(1, 0, 0),
                    bias=False)
                self.conv2 = nn.Conv3d(
                    planes,
                    planes,
                    kernel_size=(1, 3, 3),
                    stride=(self.conv2_stride_t, self.conv2_stride, self.conv2_stride),
                    padding=(0, dilation, dilation),
                    dilation=(1, dilation, dilation),
                    bias=False)
            else:
                self.conv1 = nn.Conv3d(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=(self.conv1_stride_t, self.conv1_stride, self.conv1_stride),
                    bias=False)
                self.conv2 = nn.Conv3d(
                    planes,
                    planes,
                    kernel_size=3,
                    stride=(self.conv2_stride_t, self.conv2_stride, self.conv2_stride),
                    padding=(1, dilation, dilation),
                    dilation=(1, dilation, dilation),
                    bias=False)
        else:
            self.conv1 = nn.Conv3d(
                inplanes,
                planes,
                kernel_size=1,
                stride=(1, self.conv1_stride, self.conv1_stride),
                bias=False)
            self.conv2 = nn.Conv3d(
                planes,
                planes,
                kernel_size=(1, 3, 3),
                stride=(1, self.conv2_stride, self.conv2_stride),
                padding=(0, dilation, dilation),
                dilation=(1, dilation, dilation),
                bias=False)

        self.bn1 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.spatial_tride = spatial_stride
        self.temporal_tride = temporal_stride
        self.dilation = dilation
        self.with_cp = with_cp

        if if_nonlocal and nonlocal_cfg is not None:
            nonlocal_cfg_ = nonlocal_cfg.copy()
            nonlocal_cfg_['in_channels'] = planes * self.expansion
            self.nonlocal_block = build_nonlocal_block(nonlocal_cfg_)
        else:
            self.nonlocal_block = None

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        if self.nonlocal_block is not None:
            out = self.nonlocal_block(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   spatial_stride=1,
                   temporal_stride=1,
                   dilation=1,
                   style='pytorch',
                   inflate_freq=1,
                   inflate_style='3x1x1',
                   nonlocal_freq=1,
                   nonlocal_cfg=None,
                   down_size=None,
                   with_cp=True):
    inflate_freq = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * blocks
    nonlocal_freq = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq,) * blocks
    assert len(inflate_freq) == blocks
    assert len(nonlocal_freq) == blocks
    downsample = None
    if down_size:
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(temporal_stride, spatial_stride, spatial_stride),
                    bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
    else:
        inplanes = 1024
        planes = 2048
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=(temporal_stride, spatial_stride, spatial_stride),
                    bias=False),
                nn.BatchNorm3d(planes),
            )

    layers = []
    if down_size:
        layers.append(
            block(
                inplanes,
                planes,
                spatial_stride,
                temporal_stride,
                dilation,
                downsample,
                style=style,
                if_inflate=(inflate_freq[0] == 1),
                inflate_style=inflate_style,
                if_nonlocal=(nonlocal_freq[0] == 1),
                nonlocal_cfg=nonlocal_cfg,
                with_cp=with_cp))
    else:
        inplanes = 1024
        planes = 512
        layers.append(
            block(
                inplanes,
                planes,
                spatial_stride,
                temporal_stride,
                dilation,
                downsample,
                style=style,
                if_inflate=(inflate_freq[0] == 1),
                inflate_style=inflate_style,
                if_nonlocal=(nonlocal_freq[0] == 1),
                nonlocal_cfg=nonlocal_cfg,
                with_cp=with_cp))

    if down_size:
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(inplanes,
                      planes,
                      1, 1,
                      dilation,
                      style=style,
                      if_inflate=(inflate_freq[i] == 1),
                      inflate_style=inflate_style,
                      if_nonlocal=(nonlocal_freq[i] == 1),
                      nonlocal_cfg=nonlocal_cfg,
                      with_cp=with_cp))

        return nn.Sequential(*layers)
    else:
        inplanes = 2048
        planes = 512
        for i in range(1, blocks):
            layers.append(
                block(inplanes,
                      planes,
                      1, 1,
                      dilation,
                      style=style,
                      if_inflate=(inflate_freq[i] == 1),
                      inflate_style=inflate_style,
                      if_nonlocal=(nonlocal_freq[i] == 1),
                      nonlocal_cfg=nonlocal_cfg,
                      with_cp=with_cp))
        return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNet_SlowFast(nn.Module):
    """ResNe(x)t_SlowFast backbone.

    Args:
        depth (int): Depth of resnet, from {50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 pretrained=None,
                 pretrained2d=True,
                 num_stages=4,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 conv1_kernel_t=5,
                 conv1_stride_t=2,
                 pool1_kernel_t=1,
                 pool1_stride_t=2,
                 style='pytorch',
                 frozen_stages=-1,
                 inflate_freq=(1, 1, 1, 1),
                 inflate_stride=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 nonlocal_stages=(-1,),
                 nonlocal_freq=(0, 1, 1, 0),
                 nonlocal_cfg=None,
                 bn_eval=False,
                 bn_frozen=False,
                 partial_bn=False,
                 nsegments=32,
                 num_classes=174,
                 with_cp=True):
        super(ResNet_SlowFast, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.inflate_freqs = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * num_stages
        self.inflate_style = inflate_style
        self.nonlocal_stages = nonlocal_stages
        self.nonlocal_freqs = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq,) * num_stages
        self.nonlocal_cfg = nonlocal_cfg
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.with_cp = with_cp

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self.conv1 = nn.Conv3d(
            3, 64, kernel_size=(conv1_kernel_t, 7, 7), stride=(conv1_stride_t, 2, 2),
            padding=((conv1_kernel_t - 1) // 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(pool1_kernel_t, 3, 3), stride=(pool1_stride_t, 2, 2),
                                    padding=(pool1_kernel_t // 2, 1, 1))


        self.bn_att = nn.BatchNorm3d(512 * 4)
        self.t_att_conv = nn.Conv3d(num_classes, 1, kernel_size=3, padding=1, bias=False)
        self.t_bn_att = nn.BatchNorm3d(1)
        self.t_att_conv2 = nn.Conv2d(nsegments, nsegments, kernel_size=1, padding=0, bias=False)
        self.t_att_gap = nn.AvgPool2d(14)
        self.fc1 = nn.Linear(nsegments, nsegments)
        self.fc2 = nn.Linear(nsegments, nsegments)
        self.t_att_conv3 = nn.Conv3d(512 * 4, int(512 * 4 / 2), kernel_size=1, padding=0, bias=False)
        self.bn_att4 = nn.BatchNorm3d(int(512 * 4 / 2))
        self.att_conv = nn.Conv3d(512 * 4, num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm3d(num_classes)
        self.att_conv2 = nn.Conv3d(num_classes, num_classes, kernel_size=1, padding=0,
                                bias=False)
        #self.att_conv3 = nn.Conv3d(num_classes, 1, kernel_size=3, padding=1,
                                #bias=False)
        self.att_conv3 = nn.Conv3d(num_classes, 1, kernel_size=1, padding=0,
                                bias=False)
        self.bn_att3 = nn.BatchNorm3d(1)
        self.att_gap = nn.AvgPool3d((1, 14, 14), stride=1)
        self.sigmoid = nn.Sigmoid()


        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i 
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=self.style,
                inflate_freq=self.inflate_freqs[i],
                inflate_style=self.inflate_style,
                nonlocal_freq=self.nonlocal_freqs[i],
                nonlocal_cfg=self.nonlocal_cfg if i in self.nonlocal_stages else None,
                down_size=True,
                with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

            if i == 3:
                spatial_stride = spatial_strides[i]
                temporal_stride = temporal_strides[i]
                dilation = dilations[i]
                planes = 64 * 2 ** i 
                res_layer = make_res_layer(
                    self.block,
                    self.inplanes,
                    planes,
                    num_blocks,
                    spatial_stride=1,
                    temporal_stride=temporal_stride,
                    dilation=dilation,
                    style=self.style,
                    inflate_freq=self.inflate_freqs[i],
                    inflate_style=self.inflate_style,
                    nonlocal_freq=self.nonlocal_freqs[i],
                    nonlocal_cfg=self.nonlocal_cfg if i in self.nonlocal_stages else None,
                    down_size=False,
                    with_cp=with_cp)
                self.inplanes = planes * self.block.expansion
                layer_name = 'att_layer{}'.format(i + 1)
                self.add_module(layer_name, res_layer)
                self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2 ** (
                len(self.stage_blocks) - 1)


    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            if self.pretrained2d:
                resnet2d = ResNet(self.depth)
                load_checkpoint(resnet2d, self.pretrained, strict=False, logger=logger)
                #print(resnet2d.layer4[2].conv2.weight)
                for name, module in self.named_modules():
                    if isinstance(module, NonLocalModule):
                        module.init_weights()
                    elif isinstance(module, nn.Conv3d) and rhasattr(resnet2d, name):
                        new_weight = rgetattr(resnet2d, name).weight.data.unsqueeze(2).expand_as(module.weight) / \
                                     module.weight.data.shape[2]
                        module.weight.data.copy_(new_weight)
                        logging.info("{}.weight loaded from weights file into {}".format(name, new_weight.shape))
                        if hasattr(module, 'bias') and module.bias is not None:
                            new_bias = rgetattr(resnet2d, name).bias.data
                            module.bias.data.copy_(new_bias)
                            logging.info("{}.bias loaded from weights file into {}".format(name, new_bias.shape))
                    elif isinstance(module, nn.BatchNorm3d) and rhasattr(resnet2d, name):
                        for attr in ['weight', 'bias', 'running_mean', 'running_var']:
                            logging.info("{}.{} loaded from weights file into {}".format(name, attr, getattr(
                                rgetattr(resnet2d, name), attr).shape))
                            setattr(module, attr, getattr(rgetattr(resnet2d, name), attr))
            else:
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        res_layer = getattr(self, self.res_layers[0])
        x = res_layer(x)
        res_layer2 = getattr(self, self.res_layers[1])
        x = res_layer2(x)
        res_layer3 = getattr(self, self.res_layers[2])
        x = res_layer3(x)
        att_res_layer = getattr(self, self.res_layers[4])
        br, cr, fr, yr, xr = x.shape

        ax = self.bn_att(att_res_layer(x))
        ax = self.relu(self.bn_att2(self.att_conv(ax)))

        # ta_branch
        t_ax = self.t_att_conv(ax)
        t_ax = self.t_bn_att(t_ax)
        t_ax = self.relu(t_ax)
        ta_br, ta_cr, ta_fr, ta_yr, ta_xr = t_ax.shape

        # v2
        t_ax = t_ax.view(t_ax.size(0), t_ax.size(2), t_ax.size(3), t_ax.size(4))
        t_ax = self.t_att_conv2(t_ax)
        t_ax = self.relu(t_ax)

        t_ax = self.t_att_gap(t_ax)
        t_ax = t_ax.view(t_ax.size(0), -1)
        t_ax = self.fc1(t_ax)
        t_ax = self.relu(t_ax)
        t_ax = self.fc2(t_ax)
        t_ax = self.sigmoid(t_ax)
        t_ax = t_ax.view(t_ax.size(0), ta_cr, t_ax.size(1), 1, 1)
        self.t_att = t_ax
        
        t_rx = x * self.t_att

        # sa_branch
        self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax)))
        ax = self.att_conv2(ax)
        ax = self.att_gap(ax)

        s_rx = x * self.att
        s_rx = s_rx + x

        f_list = []
        for f in range(fr):
            t_f = t_rx[:,:,f,:,:]
            s_f = s_rx[:,:,f,:,:]
            b_list = []
            for b in range(br):
                t_b = t_f[b,:,:,:]
                s_b = s_f[b,:,:,:]
                cat_c = torch.cat((s_b,t_b), dim=0)
                b_list += [cat_c]
            cat_b = torch.stack(b_list, dim=0)
            f_list += [cat_b]
        cat_f = torch.stack(f_list, dim=2)
        rx = cat_f
        rx = self.t_att_conv3(rx)
        rx = self.relu(self.bn_att4(rx))

        res_layer4 = getattr(self, self.res_layers[3])
        rx = res_layer4(rx)
        return ax, rx, self.att, self.t_att


    def train(self, mode=True):
        super(ResNet_SlowFast, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_bn:
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                for m in mod.modules():
                    if isinstance(m, nn.BatchNorm3d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False
