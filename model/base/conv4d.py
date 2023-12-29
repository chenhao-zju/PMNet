r""" Implementation of center-pivot 4D convolution """

import torch
import torch.nn as nn


class CenterPivotConv4d(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               bias=bias, padding=padding[:2])
        # print(self.conv1.weight.shape)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])
        # print(self.conv2.weight.shape)

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        # print(out1.shape)
        bsz, inch, ha, wa, hb, wb = out1.size()
        out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        # print(self.conv1.weight.shape)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        bsz, inch, ha, wa, hb, wb = x.size()
        out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2 = self.conv2(out2)
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()

        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
            out2 = out2.squeeze()

        y = out1 + out2
        return y


class CenterPivotConv6d(nn.Module):
    r""" CenterPivot 6D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv6d, self).__init__()

        kernel_size1 = [1] + kernel_size[:2]
        stride1 = [1] + stride[:2]
        padding1 = [0] + padding[:2]
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size1, stride=stride1,
                               bias=bias, padding=padding1)
        # print(self.conv1.weight.shape)
        kernel_size2 = [1] + kernel_size[2:]
        stride2 = [1] + stride[2:]
        padding2 = [0] + padding[2:]
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size2, stride=stride2,
                               bias=bias, padding=padding2)
        # print(self.conv2.weight.shape)

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        # print(out1.shape)
        bsz, inch, ha, wa, hb, wb = out1.size()
        out1 = out1.permute(0, 1, 4, 5, 2, 3).contiguous().view(bsz, inch, -1, ha, wa)
        # print(self.conv1.weight.shape)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(1), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, outch, hb, wb, o_ha, o_wa).permute(0, 1, 4, 5, 2, 3).contiguous()

        bsz, inch, ha, wa, hb, wb = x.size()
        out2 = x.permute(0, 1, 2, 3, 4, 5).contiguous().view(bsz, inch, -1, hb, wb)
        out2 = self.conv2(out2)
        outch, o_hb, o_wb = out2.size(1), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, outch, ha, wa, o_hb, o_wb).permute(0, 1, 2, 3, 4, 5).contiguous()

        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
            out2 = out2.squeeze()

        y = out1 + out2
        return y


def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4, type='4dconv'):
    assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

    building_block_layers = []
    for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
        inch = in_channel if idx == 0 else out_channels[idx - 1]
        ksz4d = [ksz,] * 4
        str4d = [stride,] * 4
        pad4d = [ksz // 2,] * 4

        if type == '4dconv':
            building_block_layers.append(CenterPivotConv4d(inch, outch, ksz4d, str4d, pad4d))
        elif type == '6dconv':
            building_block_layers.append(CenterPivotConv6d(inch, outch, ksz4d, str4d, pad4d))
        building_block_layers.append(nn.GroupNorm(group, outch))
        building_block_layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*building_block_layers)


if __name__ == '__main__':
    inputs = torch.randn([12, 3, 24, 24, 24, 24])
    model = CenterPivotConv6d(in_channels=3, out_channels=16, kernel_size=[3,3,3,3], stride=[1,1,3,3], padding=[1,1,1,1], bias=True)

    outputs = model(inputs)
    print(outputs.shape)