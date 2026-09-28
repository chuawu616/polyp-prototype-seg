import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []
        # 1. 1x1 卷積
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        # 2. 不同 rate 的 3x3 空洞卷積
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 3. 全局平均池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 融合層
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3PlusDecoder(nn.Module):
    """
    適配 PVTv2 的 DeepLabV3+ 風格 Decoder。
    """
    def __init__(self, encoder_channels, decoder_channels=256, atrous_rates=[6, 12, 18]):
        """
        Args:
            encoder_channels (list): PVTv2 輸出的通道數列表 [64, 128, 320, 512]
            decoder_channels (int): Decoder 輸出的特徵維度
        """
        super(DeepLabV3PlusDecoder, self).__init__()
        
        # 假設 encoder_channels = [c1, c2, c3, c4]
        # 我們主要使用 c4 (深層語義) 和 c1 (淺層細節)
        self.in_channels = encoder_channels[-1] # c4
        self.low_level_channels = encoder_channels[0] # c1
        
        # ASPP 模塊處理深層特徵
        self.aspp = ASPP(self.in_channels, decoder_channels, atrous_rates)
        
        # 淺層特徵投影 (減少通道數，通常設為 48 或與 decoder_channels 相同)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(self.low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # 最終融合層
        # 輸入通道 = ASPP輸出(256) + 淺層投影(48)
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels + 48, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        
        self.out_channels = decoder_channels

    def forward(self, features_list):
        """
        Args:
            features_list: [c1, c2, c3, c4]
        """
        low_level_feat = features_list[0] # c1
        x = features_list[-1] # c4
        
        # 1. ASPP 處理深層特徵
        x = self.aspp(x) # (B, 256, H/32, W/32)
        
        # 2. 上採樣到淺層特徵尺寸
        x = F.interpolate(x, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False) # (B, 256, H/4, W/4)
        
        # 3. 處理淺層特徵
        low_level_feat = self.low_level_conv(low_level_feat) # (B, 48, H/4, W/4)
        
        # 4. 拼接
        x = torch.cat([x, low_level_feat], dim=1) # (B, 304, H/4, W/4)
        
        # 5. 融合
        x = self.final_conv(x) # (B, 256, H/4, W/4)
        
        return x