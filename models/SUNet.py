import torch.nn as nn
from models.SUNet_detail import SUNet


class SUNet_model(nn.Module):
    def __init__(self, img_size=256, embed_dim=96, depth=[8, 8, 8, 8], num_heads=[8, 8, 8, 8], window_size=8,
                mlp_ratio=4., qk_scale=8):
        super(SUNet_model, self).__init__()
        # self.config = config
        self.swin_unet = SUNet(img_size=img_size,
                               patch_size=4,
                               in_chans=3,
                               out_chans=3,
                               embed_dim=embed_dim,
                               depths=depth,
                               num_heads=num_heads,
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=True,
                               qk_scale=qk_scale,
                               drop_rate=0.,
                               drop_path_rate=0.1,
                               ape=False,
                               patch_norm=True,
                               use_checkpoint=False)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits
    
if __name__ == '__main__':
    from utils.model_utils import network_parameters
    import torch
    import yaml
    from thop import profile
    from utils.model_utils import network_parameters

    ## Load yaml configuration file
    with open('../training.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    Train = opt['TRAINING']
    OPT = opt['OPTIM']

    height = 256
    width = 256
    x = torch.randn((1, 156, height, width))  # .cuda()
    model = SUNet_model(opt)  # .cuda()
    out = model(x)
    flops, params = profile(model, (x,))
    print(out.size())
    print(flops)
    print(params)
