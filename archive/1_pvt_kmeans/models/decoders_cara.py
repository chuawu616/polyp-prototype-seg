# 檔案: models/decoders_cara.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 基礎卷積塊 ---
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

# --- CFP: Context Feature Pyramid (上下文特徵金字塔) ---
# 用於處理最深層特徵，捕捉多尺度上下文
class CFPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFPModule, self).__init__()
        # 擴張率分別為 1, 2, 4, 8
        self.dilations = [1, 2, 4, 8]
        self.convs = nn.ModuleList()
        for d in self.dilations:
            self.convs.append(BasicConv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d))
        
        self.fusion = BasicConv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        outs = []
        for conv in self.convs:
            outs.append(conv(x))
        # 拼接並融合
        out = torch.cat(outs, dim=1)
        out = self.fusion(out)
        return out

# --- CRA: Channel-wise Reverse Attention (通道反向注意力) ---
# 這是 CaraNet 的靈魂，用於在解碼過程中細化邊界
class CRAModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CRAModule, self).__init__()
        # 用於生成注意力圖的卷積
        self.conv_atten = nn.Sequential(
            BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        # 用於融合輸入特徵的卷積
        self.conv_feat = BasicConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_out = BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, prev_mask=None):
        """
        x: 當前層級的特徵
        prev_mask: 上一層生成的預測 Mask (經過上採樣)
        """
        # 1. 特徵變換
        feat = self.conv_feat(x)
        
        if prev_mask is not None:
            # 2. 反向注意力機制
            # 我們希望關注那些【還沒有被預測為前景】的區域，或者是邊界區域
            # prev_mask 是 sigmoid 後的概率
            # reverse_attention_weight = 1 - prev_mask
            atten = torch.sigmoid(-prev_mask) # 負號 + sigmoid 等效於反轉關注點
            
            # 3. 加權
            feat = feat * atten
        
        # 4. 輸出細化後的特徵
        out = self.conv_out(feat)
        
        # 5. 同時生成當前層級的預測 Mask (用於下一層的反向注意力)
        # 注意：這裡返回的是 Logits，下一層使用時需要 Sigmoid
        mask_logits = self.conv_atten(x)
        
        return out, mask_logits

class CaraNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=64):
        super(CaraNetDecoder, self).__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        # CFP 處理最深層 (Stage 4)
        # 輸入: encoder_channels[3] (512)
        self.cfp = CFPModule(encoder_channels[3], decoder_channels)
        
        # 轉換層: 只對前三個淺層特徵進行降維
        # c4 不在這裡降維，而是通過 CFP 處理
        self.trans_convs = nn.ModuleList([
            BasicConv2d(c, decoder_channels, kernel_size=1) 
            for c in encoder_channels[:-1] # 只取前三個: [64, 128, 320]
        ])

        # CRA 模塊 (保持不變)
        self.cra3 = CRAModule(decoder_channels * 2, decoder_channels)
        self.cra2 = CRAModule(decoder_channels * 2, decoder_channels)
        self.cra1 = CRAModule(decoder_channels * 2, decoder_channels)

    def forward(self, features_list):
        # features_list: [c1, c2, c3, c4]
        
        # 1. 處理前三個淺層特徵 (降維)
        # zip 會自動匹配最短的列表，所以只處理前三個
        trans_feats = [conv(f) for conv, f in zip(self.trans_convs, features_list[:-1])]
        c1, c2, c3 = trans_feats
        
        # 取出最深層特徵 (保持原始維度)
        c4 = features_list[-1]
        
        # 2. CFP 處理最深層
        # 輸入: (B, 512, H/32, W/32)
        # 輸出: (B, 64, H/32, W/32)
        c4_feat = self.cfp(c4) 
        
        # 3. 逐層融合 (保持不變)
        # --- Stage 3 ---
        c4_up = F.interpolate(c4_feat, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        feat3 = torch.cat([c3, c4_up], dim=1)
        out3, mask3 = self.cra3(feat3, prev_mask=None)
        
        # --- Stage 2 ---
        out3_up = F.interpolate(out3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        mask3_up = F.interpolate(mask3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        feat2 = torch.cat([c2, out3_up], dim=1)
        out2, mask2 = self.cra2(feat2, prev_mask=mask3_up)
        
        # --- Stage 1 ---
        out2_up = F.interpolate(out2, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        mask2_up = F.interpolate(mask2, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        feat1 = torch.cat([c1, out2_up], dim=1)
        out1, mask1 = self.cra1(feat1, prev_mask=mask2_up)
        
        return out1