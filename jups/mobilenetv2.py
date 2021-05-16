import torch.nn as nn
# from enot.models import register_searchable_op
# from searchable_mib import SearchableMobileInvertedBottleneck

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, padding = None, stride = 1, dilation = 1, kernel_size = 3, groups = 1):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super().__init__(
                nn.Conv2d(in_channels, out_channels, 
                          kernel_size = kernel_size,
                          stride = stride,
                          padding = padding, 
                          dilation = dilation, 
                          groups = groups,
                          bias = False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace = True)
         )

# @register_searchable_op("dilMIB")
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation, padding, expand_ratio, kernel_size = 3):
        super().__init__()
        mid_channels = int(round(in_channels * expand_ratio))
        self.use_skip_connection = stride == 1 and in_channels == out_channels
        layers = []
        
        if expand_ratio > 1:
            layers.append(ConvBNReLU(in_channels, mid_channels, kernel_size = 1))
            
        layers.extend([
            ConvBNReLU(mid_channels, mid_channels, 
                       stride = stride, 
                       padding = padding, 
                       dilation = dilation,
                       groups = mid_channels),
            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        y = self.conv(x)
        return x + y if self.use_skip_connection else y
    
class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 32
        layers = [
            ConvBNReLU(3, in_channels, stride = 2)
        ]
        
        inverted_residual_setting = [
            # t, c,  n, s, p, d
            [1, 16, 1, 1, 1, 1],

            [6, 24, 1, 2, 1, 1],
            [6, 24, 1, 1, 1, 1],

            [6, 32, 1, 2, 1, 1],
            [6, 32, 1, 1, 1, 1],
            [6, 32, 1, 1, 1, 1],

            [6, 64, 1, 1, 2, 2],
            [6, 64, 1, 1, 1, 1],
            [6, 64, 1, 1, 1, 1],
            [6, 64, 1, 1, 1, 1],

            [6, 96, 1, 1, 1, 1],
            [6, 96, 1, 1, 1, 1],
            [6, 96, 1, 1, 1, 1],
        ]
        
        for t, c, n, s, p, d in inverted_residual_setting:
            for i in range(n):
                out_channels = c
                layers.append(
                    InvertedResidual(in_channels, out_channels, 
                                     stride = s if i == 0 else 1, 
                                     padding = p, 
                                     dilation = d, 
                                     expand_ratio = t)
                )
                in_channels = out_channels
        
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

    

class SearchableMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                    ConvBNReLU(3, 32, 2),
            
             self.search_block(32, 16),

             self.search_block(16, 24, 2),
             self.search_block(24, 24),

             self.search_block(24, 32, 2),
             self.search_block(32, 32),
             self.search_block(32, 32),

             self.search_block(32, 64, 1, 2, 2),
             self.search_block(64, 64),
             self.search_block(64, 64),
             self.search_block(64, 64),

             self.search_block(64, 96),
             self.search_block(96, 96),
             self.search_block(96, 96),
        )
        
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
    
    @staticmethod
    def search_block(in_channels, out_channels, stride = 1, padding = 1, dilation = 1):
        blocks = [
            SearchableMobileInvertedBottleneck(in_channels, out_channels,
                                                 expand_ratio = 1,
                                                 stride = stride,
                                                 dilation = dilation,
                                                 padding = padding),
            SearchableMobileInvertedBottleneck(in_channels, out_channels,
                                                 expand_ratio = 3,
                                                 stride = stride,
                                                 dilation = dilation,
                                                 padding = padding),
            SearchableMobileInvertedBottleneck(in_channels, out_channels,
                                                 expand_ratio = 6,
                                                 stride = stride,
                                                 dilation = dilation,
                                                 padding = padding),
            ]
        return SearchVariantsContainer(blocks)
    
    def forward(self, x):
        return self.model(x)