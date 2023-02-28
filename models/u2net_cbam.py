import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import RSU, RSU4F, ConvBNReLU, DownConvBNReLU, UpConvBNReLU
from models.cbam import CBAM, ChannelAttentionModule

class U2netLiteCbam2Generator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc: int, output_nc: int, norm_layer=nn.BatchNorm2d):
        """Construct a U2net generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            norm_layer      -- normalization layer
        """
        super(U2netLiteCbam2Generator, self).__init__()
        self.model = u2net_lite_cbam2(input_nc, output_nc, norm_layer)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


def u2net_lite_cbam2(input_nc: int, output_nc: int, norm_layer=nn.BatchNorm2d):
    cfg = {
        # height, input_nc, mid_ch, output_nc, RSU4F, side
        "encode": [[7, input_nc, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, input_nc, mid_ch, output_nc, RSU4F, side
        "decode": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return U2Net_cbam2(cfg, output_nc, norm_layer)

class U2Net_cbam2(nn.Module):
    def __init__(self, cfg: dict, output_nc: int, norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert "encode" in cfg
        assert "decode" in cfg
        self.encode_num = len(cfg["encode"])

        encode_list = []
        side_list = []
        cbam_list = []
        for c in cfg["encode"]:
            # c: [height, input_nc, mid_ch, output_nc, RSU4F, side]
            assert len(c) == 6
            encode_list.append(RSU(*c[:4], norm_layer=norm_layer) if c[4] is False else RSU4F(*c[1:4], norm_layer=norm_layer))
            cbam_list.append(ChannelAttentionModule(c[3]))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], output_nc, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        decode_list = []
        for c in cfg["decode"]:
            # c: [height, input_nc, mid_ch, output_nc, RSU4F, side]
            assert len(c) == 6
            decode_list.append(RSU(*c[:4], norm_layer=norm_layer) if c[4] is False else RSU4F(*c[1:4], norm_layer=norm_layer))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], output_nc, kernel_size=3, padding=1))
        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        self.cbam_list = nn.ModuleList(cbam_list)
        self.out_conv = nn.Conv2d(self.encode_num * output_nc, output_nc, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        # collect encode outputs
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            x = self.cbam_list[i](x) + x
            encode_outputs.append(x)
            if i != self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        # collect decode outputs
        x = encode_outputs.pop()
        decode_outputs = [x]
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x = m(torch.cat([x, x2], dim=1))
            decode_outputs.insert(0, x)

        # collect side outputs
        side_outputs = []
        for m in self.side_modules:
            x = decode_outputs.pop()
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
            side_outputs.insert(0, x)

        x = self.out_conv(torch.cat(side_outputs, dim=1))

        # if self.training:
        #     # do not use torch.sigmoid for amp safe
        #     return [x] + side_outputs
        # else:
        #     return torch.sigmoid(x)
        tanh = nn.Tanh()
        return tanh(x)