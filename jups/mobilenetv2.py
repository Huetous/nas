import torch.nn as nn
# from enot.models import register_searchable_op

def _make_divisible(v, divisor, min_value = None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, dilation = 1, groups = 1):
        padding = (kernel_size - 1) // 2 * dilation
        super().__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                          dilation = dilation, groups = groups, bias = False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace = True)
         )

# short_args = {
#     "k": ("kernel_size", int),
#     "t": ("expand_ratio", int)
# }
            
# @register_searchable_op("MIB", short_args)
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio = 1):
        super().__init__()
        mid_channels = int(round(in_channels * expand_ratio))
        self.use_skip_connection = stride == 1 and in_channels == out_channels
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
        y = self.conv(x)
        return x + self.conv(x) if self.use_skip_connection else self.conv(x)
    
class MobileNetV2(nn.Module):
    def __init__(self, in_channels, last_channels, num_classes = 1000, width_mult = 1.0, inverted_residual_setting = None,
                 round_nearest= 8, include_classifier = False):
        super().__init__()

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

        in_channels = _make_divisible(in_channels * width_mult, round_nearest)
        last_channels = _make_divisible(last_channels * max(1.0, width_mult), round_nearest)
        
        layers = [
            ConvBNReLU(3, in_channels, stride = 2)
        ]

        for t, c, n, s in inverted_residual_setting:
            out_channels = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                layers.append(
                    InvertedResidual(in_channels, out_channels, s if i == 0 else 1, expand_ratio = t)
                )
                in_channels = out_channels
        
        if include_classifier:
            layers.extend([
                ConvBNReLU(in_channels, last_channels, kernel_size=1),
                nn.AdaptivePool2d(1),
                nn.Dropout(0.2),
                nn.Linear(last_channels, num_classes)
            ])
        
        self.model = nn.Sequential(*layers)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)
