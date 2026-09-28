import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange, repeat
from timm.layers import trunc_normal_

from lib.pvtv2 import pvt_v2_b2
from lib.decoders import CASCADE
from lib.contrast import momentum_update, l2_normalize
from lib.sinkhorn import distributed_sinkhorn

class Prototype_CASCADE_v3(nn.Module):
    def __init__(self, num_classes=2, num_prototype=5, gamma=0.999, 
                 use_prototype=True, update_prototype=True, pretrain_prototype=False,
                 encoder_path=None, pretrained_model_path=None, kmeans_center_path=None,
                 target_node='dd2'):
        super(Prototype_CASCADE_v3, self).__init__()
        
        self.num_classes = num_classes
        self.num_prototype = num_prototype
        self.gamma = gamma
        self.use_prototype = use_prototype
        self.update_prototype = update_prototype
        self.pretrain_prototype = pretrain_prototype
        self.target_node = target_node

        self.backbone = pvt_v2_b2()

        if pretrained_model_path is None:
            path = '/home/U116med/wch_code/non_learnable/weights/pvt_v2_b2.pth'
            if encoder_path:
                path = encoder_path

            save_model = torch.load(path, weights_only=True)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
            
        self.decoder = CASCADE(channels=[512, 320, 128, 64])

        # 設定節點維度映射表
        self.channels_dict = {
            'x1': 64, 'x2': 128, 'x3': 320, 'x4': 512,
            'dd4': 512, 'dd3': 320, 'dd2': 128, 'dd1': 64, 'd1': 64
        }
        
        self.in_channels_node = self.channels_dict[self.target_node]
        self.in_channels_d1 = 64

        # True Prototype 建立在 target_node 維度上
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, self.in_channels_node),
                                       requires_grad=True)

        # 雙軌跡特徵正規化
        self.feat_norm_node = nn.LayerNorm(self.in_channels_node)
        self.feat_norm_d1 = nn.LayerNorm(self.in_channels_d1)
        self.temperature = nn.Parameter(torch.tensor(10.0), requires_grad=False)

        self.proto_proj_mlp = nn.Sequential(
            nn.Linear(self.in_channels_node, self.in_channels_node),
            nn.LayerNorm(self.in_channels_node),
            nn.GELU(),
            nn.Linear(self.in_channels_node, self.in_channels_d1)
        )

        # 處理 Prototype 初始化
        if kmeans_center_path is not None:
            self._load_kmeans_centers(kmeans_center_path)
        else:
            trunc_normal_(self.prototypes, std=0.02)

        self._init_mlp_weights()

        if pretrained_model_path is not None:
            self._load_pretrained_model(pretrained_model_path)
            
    def _init_mlp_weights(self):
        for m in self.proto_proj_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _load_pretrained_model(self, path):
        save_model = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        print(f"Loaded {len(state_dict)} shared layers from {path}.")

    def _load_kmeans_centers(self, path):
        if os.path.exists(path):
            kmeans_centers = torch.load(path, map_location='cpu')
            if kmeans_centers.shape == self.prototypes.shape:
                self.prototypes.data.copy_(kmeans_centers)
                print(f"Loaded K-Means centers from {path}.")
            else:
                print(f"Shape mismatch! Expected {self.prototypes.shape}, but got {kmeans_centers.shape}. Using trunc_normal_ instead.")
                trunc_normal_(self.prototypes, std=0.02)
        else:
            print(f"K-Means center file not found at {path}. Using trunc_normal_ instead.")
            trunc_normal_(self.prototypes, std=0.02)

    def prototype_learning(self, _c_node, _c_d1, out_seg, proj_prototypes, node_gt_label, sim_matrix_node, shape_node, shape_d1):
        b_n, h_n, w_n = shape_node
        b_d, h_d, w_d = shape_d1
        
        # 建立有效像素遮罩 (Valid Pixel Mask)
        out_seg_node = F.interpolate(out_seg, size=(h_n, w_n), mode='bilinear', align_corners=False)
        pred_seg_node = torch.max(out_seg_node, 1)[1].view(-1)
        valid_mask = (node_gt_label == pred_seg_node)

        # PPC Logits 基礎計算
        contrast_logits_node = torch.mm(_c_node, self.prototypes.view(-1, self.prototypes.shape[-1]).t())
        contrast_logits_node = contrast_logits_node * self.temperature

        # PPD Logits 基礎計算
        contrast_logits_d1 = torch.mm(_c_d1, proj_prototypes.view(-1, proj_prototypes.shape[-1]).t())
        contrast_logits_d1 = contrast_logits_d1 * self.temperature

        # 儲存 Sinkhorn 分配的類別標籤 (Proto Labels)
        proto_labels_node = node_gt_label.clone().float() 
        protos = self.prototypes.data.clone()
        
        for k in range(self.num_classes):
            # 從相似度矩陣中取出該類別的初始權重
            init_q = sim_matrix_node[..., k] 
            init_q = init_q[node_gt_label == k, ...] 
            if init_q.shape[0] == 0:
                continue 

            q, indexs = distributed_sinkhorn(init_q)

            # 過濾高置信度特徵進行更新
            m_k = valid_mask[node_gt_label == k]  
            c_k = _c_node[node_gt_label == k, ...] 

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)
            m_q = q * m_k_tile  

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            c_q = c_k * c_k_tile  

            f = m_q.transpose(0, 1) @ c_q  
            n = torch.sum(m_q, dim=0) 

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(old_value=protos[k, n != 0, :], 
                                            new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            # 將分配到的子類別 index 寫入標籤矩陣
            proto_labels_node[node_gt_label == k] = indexs.float() + (self.num_prototype * k)

        self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        # 解析度對齊：將 Target 節點的子類別標籤上採樣至 d1 階層
        proto_labels_node_spatial = proto_labels_node.view(b_n, 1, h_n, w_n)
        proto_labels_d1 = F.interpolate(proto_labels_node_spatial, size=(h_d, w_d), mode='nearest').view(-1)

        return contrast_logits_node, contrast_logits_d1, proto_labels_node, proto_labels_d1

    def forward(self, x, gt_semantic_seg=None):
        x1, x2, x3, x4 = self.backbone(x)
        outs = self.decoder(x4, [x3, x2, x1])
        
        feats = {
            'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
            'dd4': outs[0], 'dd3': outs[1], 'dd2': outs[2], 'dd1': outs[3], 'd1': outs[4]
        }
        
        node_feat = feats[self.target_node]
        d1_feat = outs[4] 

        b_node, c_node, h_node, w_node = node_feat.shape
        b_d1, c_d1, h_d1, w_d1 = d1_feat.shape
        
        # 1. Target 節點特徵前處理
        _c_node = rearrange(node_feat, 'b c h w -> (b h w) c')
        _c_node = self.feat_norm_node(_c_node)
        _c_node = l2_normalize(_c_node)
        
        # 2. D1 節點特徵前處理
        _c_d1 = rearrange(d1_feat, 'b c h w -> (b h w) c')
        _c_d1 = self.feat_norm_d1(_c_d1)
        _c_d1 = l2_normalize(_c_d1)

        # 3. 準備 True Prototype 與 Projected Prototype
        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        proj_prototypes = self.proto_proj_mlp(self.prototypes)
        proj_prototypes = l2_normalize(proj_prototypes)

        # 4. 相似度矩陣計算 (Similarity Matrix)
        sim_matrix_node = torch.einsum('nd,kmd->nmk', _c_node, self.prototypes)
        sim_matrix_d1 = torch.einsum('nd,kmd->nmk', _c_d1, proj_prototypes)

        out_seg = torch.amax(sim_matrix_d1, dim=1)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=b_d1, h=h_d1)

        sim_map_small = rearrange(sim_matrix_d1, "(b h w) m k -> b (k m) h w", b=b_d1, h=h_d1)

        logits_high = F.interpolate(out_seg, size=x.shape[-2:], mode='bilinear', align_corners=False)
        similarity_map_high = F.interpolate(sim_map_small, size=x.shape[-2:], mode='bilinear', align_corners=False)

        if self.training and self.use_prototype is True and gt_semantic_seg is not None:
            # 建立符合 Target 節點解析度的 Ground Truth 標籤
            node_gt_label = F.interpolate(gt_semantic_seg.unsqueeze(1).float(), size=(h_node, w_node), mode='nearest').view(-1)
            
            contrast_logits_node, contrast_logits_d1, proto_labels_node, proto_labels_d1 = self.prototype_learning(
                _c_node=_c_node, 
                _c_d1=_c_d1, 
                out_seg=out_seg,
                proj_prototypes=proj_prototypes, 
                node_gt_label=node_gt_label, 
                sim_matrix_node=sim_matrix_node,
                shape_node=(b_node, h_node, w_node),
                shape_d1=(b_d1, h_d1, w_d1)
            )
                
            return {
                'seg': logits_high,                   
                'logits_node': contrast_logits_node,
                'logits_d1': contrast_logits_d1,
                'labels_node': proto_labels_node,
                'labels_d1': proto_labels_d1,            
                'similarity_map': similarity_map_high 
            }
                
        return logits_high, similarity_map_high

if __name__ == '__main__':
    model = Prototype_CASCADE_v3(num_classes=2, num_prototype=5, target_node='dd2').cuda()
    dummy_input = torch.randn(2, 3, 352, 352).cuda()
    dummy_gt = torch.randint(0, 2, (2, 1, 352, 352)).cuda()
    
    model.train()
    out_dict = model(dummy_input, dummy_gt)
    print("Training Seg Shape:", out_dict['seg'].shape)
    print("Training Sim Map Shape:", out_dict['similarity_map'].shape)
    print("Training Contrast Logits (Node) Shape:", out_dict['logits_node'].shape)
    print("Training Contrast Labels (Node) Shape:", out_dict['labels_node'].shape)
    
    model.eval()
    logits, sim_map = model(dummy_input)
    print("Eval Logits Shape:", logits.shape)
    print("Eval Sim Map Shape:", sim_map.shape)