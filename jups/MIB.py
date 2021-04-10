import torch.nn as nn
from enot.models import register_searchable_op

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, dilation = 1, groups = 1):
         super().__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                          dilation = dilation, groups = groups, bias = False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace = True)
         )

short_args = {
    "k": ("kernel_size", int),
    "t": ("expand_ratio", int)
}
            
@register_searchable_op("MIB", short_args)
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_skip_connection, expand_ratio = 1):
        super().__init__()
        mid_channels = int(round(in_channels * expand_ratio))
        self.use_skip_connection = use_skip_connection and in_channels == out_channels
        layers = []
        
        if expand_ratio > 1:
            layers.append(ConvBNReLU(in_channels, mid_channels, kernel_size = 1))
        
        layers.extend([
            ConvBNReLU(mid_channels, mid_channels, stride = stride, groups = mid_channels),
            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_skip_connection else self.conv(x)