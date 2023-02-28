import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class ConvBNReLU(nn.Module):
    def __init__(self, input_nc: int, output_nc: int, kernel_size: int = 3, dilation: int = 1, norm_layer=nn.BatchNorm2d):
        super().__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size, padding=padding, dilation=dilation, bias=use_bias)
        self.bn = nn.BatchNorm2d(output_nc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):
    def __init__(self, input_nc: int, output_nc: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True, norm_layer=nn.BatchNorm2d):
        super().__init__(input_nc, output_nc, kernel_size, dilation, norm_layer=norm_layer)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, input_nc: int, output_nc: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True, norm_layer=nn.BatchNorm2d):
        super().__init__(input_nc, output_nc, kernel_size, dilation, norm_layer=norm_layer)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


class RSU(nn.Module):
    def __init__(self, height: int, input_nc: int, mid_ch: int, output_nc: int, norm_layer=nn.BatchNorm2d):
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNReLU(input_nc, output_nc)

        encode_list = [DownConvBNReLU(output_nc, mid_ch, flag=False, norm_layer=norm_layer)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False, norm_layer=norm_layer)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch, norm_layer=norm_layer))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else output_nc, norm_layer=norm_layer))

        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2, norm_layer=norm_layer))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in

class RSU4F(nn.Module):
    def __init__(self, input_nc: int, mid_ch: int, output_nc: int, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv_in = ConvBNReLU(input_nc, output_nc, norm_layer=norm_layer)
        self.encode_modules = nn.ModuleList([ConvBNReLU(output_nc, mid_ch, norm_layer=norm_layer),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2, norm_layer=norm_layer),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4, norm_layer=norm_layer),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8, norm_layer=norm_layer)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4, norm_layer=norm_layer),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2, norm_layer=norm_layer),
                                             ConvBNReLU(mid_ch * 2, output_nc, norm_layer=norm_layer)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in
