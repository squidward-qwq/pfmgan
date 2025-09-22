import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, dem_channels=1, condition_channels=3):
        super(Discriminator, self).__init__()
        input_concat_channels = dem_channels + condition_channels
        def d_block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if normalize: layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *d_block(input_concat_channels, 64, normalize=False),
            *d_block(64, 128),
            *d_block(128, 256),
            *d_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, dem_image, condition_image):
        combined_input = torch.cat((dem_image, condition_image), dim=1)
        return self.model(combined_input)