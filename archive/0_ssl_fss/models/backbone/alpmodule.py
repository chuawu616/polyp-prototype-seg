"""
ALPModule
"""
import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np

class MultiProtoAsConv(nn.Module):
    def __init__(self, proto_grid, feature_hw, upsample_mode='bilinear'):
        super(MultiProtoAsConv, self).__init__()
        self.proto_grid = proto_grid
        self.upsample_mode = upsample_mode
        #決定輸出的prototype map大小對應的pooling kernel size
        kernel_size = [ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)]
        self.avg_pool_op = nn.AvgPool2d(kernel_size)

    def forward(self, qry, sup_x, sup_y, mode, thresh, isval=False, val_wsize=None, vis_sim=False, **kwargs):
        # qry:   [1, 1, n_queries, C, H', W']
        # sup_x: [1, n_shots, 1, C, H', W']
        # sup_y: [1, n_shots, 1, 1, H', W']

        # .squeeze(0) 移除 way 維度, .squeeze(0) 移除 nb 維度
        qry = qry.squeeze(0).squeeze(0) #(n_queries, C, H', W')
        sup_x = sup_x.squeeze(0).squeeze(1) #(n_shots, C, H', W')
        sup_y = sup_y.squeeze(0).squeeze(1) #(n_shots, 1, H', W')

        #原model用這個，但F.normalize似乎差異不大
        def safe_norm(x, p = 2, dim = 1, eps = 1e-4): #對channel維度取norm
            x_norm = torch.norm(x, p = p, dim = dim) # .detach()
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x
            
        if mode == 'mask': #class level prototype only
            #取出mask框出的部分後，根據mask內有多少pixel取平均
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5) #nb x C
            #所有的class prototype被取平均
            proto = proto.mean(dim=0, keepdim=True) #1 x C
            #proto[..., None, None] 把資料轉成 1 x C x 1 x 1
            #產生一張similarity mask
            # x 20.0是為了放大logits分數
            pred_mask = F.cosine_similarity(qry, proto[..., None, None], dim=1, eps=1e-4) * 20.0
            vis_dict = {'proto_assign': None} #things to visualize
            if vis_sim:
                vis_dict['raw_local_sims'] = pred_mask
            return pred_mask.unsqueeze(1), [pred_mask], vis_dict

        elif mode == 'gridconv' or mode == 'gridconv+':
            sup_nshot, nch, _, _ = sup_x.shape

            #print("\n--- ALPModule DEBUG ---")
            #print(f"Query shape for conv2d: {qry.shape}")
            # print(f"sup_x initial shape: {sup_x.shape}")
            # print(f"sup_y initial shape: {sup_y.shape}")
            # print(f"Channel count (nch): {nch}")
            
            pooled_sup_x = self.avg_pool_op(sup_x)
            pooled_sup_y = self.avg_pool_op(sup_y)

            flat_sup_x = pooled_sup_x.permute(0, 2, 3, 1).contiguous().view(-1, nch)
            flat_sup_y = pooled_sup_y.view(-1)

            local_protos = flat_sup_x[flat_sup_y > thresh]

            if mode == 'gridconv+':
                glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
                all_protos = torch.cat([local_protos, glb_proto], dim=0)
            else:
                all_protos = local_protos

            if all_protos.shape[0] == 0:
                return torch.zeros_like(qry[:, :1, :, :]), [None], {'proto_assign': None}

            # 我們應該在 channel 維度 (dim=1) 上進行歸一化
            pro_n = F.normalize(all_protos, p=2, dim=1)
            qry_n = F.normalize(qry, p=2, dim=1)
            #pro_n.unsqueeze(-1).unsqueeze(-1) 與 proto[..., None, None]等效
            dists = F.conv2d(qry_n, pro_n.unsqueeze(-1).unsqueeze(-1)) * 20.0
            
            pred_grid = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)

            debug_assign = dists.argmax(dim=1).float().detach() 
            #detach表示不參與back propagation 加不加沒影響，argmax不可微
            #debug_assign用來看最後每個pixel被分到哪類
            vis_dict = {'proto_assign': debug_assign}
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict
        else:
            raise NotImplementedError
    def batch_forward(self, qry_batch, sup_x_batch, sup_y_batch, mode, thresh, **kwargs):
        """
        处理一个批次的 support-query 对。
        
        Args:
            qry_batch (Tensor): (B, C, H, W)
            sup_x_batch (Tensor): (B, C, H, W)
            sup_y_batch (Tensor): (B, 1, H, W)
        """
        batch_size, nch, _, _ = qry_batch.shape

        # 1. 池化
        pooled_sup_x = self.avg_pool_op(sup_x_batch) # (B, C, H_p, W_p)
        pooled_sup_y = self.avg_pool_op(sup_y_batch) # (B, 1, H_p, W_p)
        
        # 2. 展平
        flat_sup_x = pooled_sup_x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, nch) # (B, N_proto, C)
        flat_sup_y = pooled_sup_y.view(batch_size, -1) # (B, N_proto)

        # 3. 筛选、计算、匹配 (这部分无法简单地批处理，因为每个样本的原型数量不同)
        # 我们仍然需要一个 for 循环，但这比在 grid_proto_fewshot.py 中循环更高效
        
        all_scores = []
        for i in range(batch_size):
            # 获取当前样本的数据
            sup_x_i, sup_y_i = sup_x_batch[i], sup_y_batch[i]
            flat_x_i, flat_y_i = flat_sup_x[i], flat_sup_y[i]
            qry_i = qry_batch[i].unsqueeze(0) # (1, C, H, W)
            
            # 筛选局部原型
            local_protos = flat_x_i[flat_y_i > thresh]

            # 组合原型
            if mode == 'gridconv+':
                glb_proto = torch.sum(sup_x_i * sup_y_i, dim=(-1, -2)) / (sup_y_i.sum(dim=(-1, -2)) + 1e-5)
                all_protos = torch.cat([local_protos, glb_proto.unsqueeze(0)], dim=0) if local_protos.shape[0] > 0 else glb_proto.unsqueeze(0)
            else:
                all_protos = local_protos

            if all_protos.shape[0] == 0:
                # 如果没有原型，添加一个零分数的占位符
                all_scores.append(torch.zeros_like(qry_i[:, :1, :, :]))
                continue
                
            # 匹配
            pro_n = F.normalize(all_protos, p=2, dim=1)
            qry_n = F.normalize(qry_i, p=2, dim=1)
            dists = F.conv2d(qry_n, pro_n.unsqueeze(-1).unsqueeze(-1)) * 20.0
            score = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)
            all_scores.append(score)

        final_scores = torch.cat(all_scores, dim=0) # (B, 1, H, W)
        
        # 返回分数，以及用于调试的 None
        return final_scores, None, None