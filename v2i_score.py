
import torch
import torch.nn as nn

class HardConstraintOutputLayer(nn.Module):
    def __init__(
        self,
        score_min=0.0,
        score_max=1.0,
        conf_w_min=0.1,
        conf_w_max=1.0,
        delay_thresh=150,
        iou_thresh=0.1
    ):
        super().__init__()
        self.score_min = score_min
        self.score_max = score_max
        self.conf_w_min = conf_w_min
        self.conf_w_max = conf_w_max
        self.delay_thresh = delay_thresh
        self.iou_thresh = iou_thresh

    def forward(self, veh2inf_attn, inf2veh_attn, veh_targets, inf_targets, iou_3d_matrix):
        """
        硬约束辅助决策输出层
        输入：
            veh2inf_attn: [B, heads, Nv, Ni] 车端关注路端的注意力权重
            inf2veh_attn: [B, heads, Ni, Nv] 路端关注车端的注意力权重
            veh_targets: [B, Nv, 10] 车端目标 [x,y,z,w,l,h,yaw,score,cls,timestamp]
            inf_targets: [B, Ni, 10] 路端目标
            iou_3d_matrix: [B, Nv, Ni] 3D IoU矩阵
        输出：
            字典：关联度分数、路端置信度权重、异常标记、安全标志
        """
        B, heads, Nv, Ni = veh2inf_attn.shape

        # ---------------------- 1. 关联度分数（逐头计算） ----------------------
        # 交叉注意力权重对称化
        attn_fused = (veh2inf_attn + inf2veh_attn.transpose(2, 3)) * 0.5
        # IoU自适应权重：IoU高于阈值时线性衰减
        iou_weight = torch.where(
            iou_3d_matrix.unsqueeze(1) < self.iou_thresh,
            torch.ones_like(iou_3d_matrix.unsqueeze(1)),
            1.0 - (iou_3d_matrix.unsqueeze(1) - self.iou_thresh) / (1.0 - self.iou_thresh)
        )
        attn_fused = attn_fused * iou_weight
        association_score = torch.clamp(attn_fused, self.score_min, self.score_max)

        # ---------------------- 2. 路端置信度权重（聚合多头和车端） ----------------------
        # 提取路端目标属性
        inf_score = inf_targets[..., 7]      # [B, Ni]
        inf_delay = inf_targets[..., 9]      # [B, Ni]
        delay_penalty = torch.exp(-inf_delay / self.delay_thresh)  # [B, Ni]

        # 聚合车端对路端的注意力：在头部和车端维度上取平均，得到每个路端被关注的整体强度
        inf_attn = veh2inf_attn.mean(dim=(1, 2))  # [B, Ni]

        # 计算原始置信度权重（未 clamp）
        conf_weight_raw = inf_attn * inf_score * delay_penalty  # [B, Ni]

        # 异常检测：基于原始值判断是否越界（在 clamp 之前）
        weight_outlier = ((conf_weight_raw < self.conf_w_min) | (conf_weight_raw > self.conf_w_max)).float()
        anomaly_flags = {}
        anomaly_flags["weight_outlier"] = weight_outlier  # [B, Ni]

        # 应用 clamp 得到最终权重
        infra_conf_weight = torch.clamp(conf_weight_raw, self.conf_w_min, self.conf_w_max)

        # ---------------------- 3. 异常标记（逐头检测） ----------------------
        # 异常匹配：高关联度但低IoU
        abnormal_match_per_head = ((association_score > 0.5) & (iou_3d_matrix.unsqueeze(1) < self.iou_thresh)).float()
        anomaly_flags["abnormal_match"] = abnormal_match_per_head.sum(dim=1)  # [B, Nv, Ni]，统计异常头数

        # 路端异常目标：延迟过高或置信度过低
        anomaly_flags["infra_abnormal_target"] = ((inf_delay > self.delay_thresh) | (inf_score < 0.2)).float()

        # ---------------------- 4. 安全判断 ----------------------
        # 存在任意异常标记（匹配异常、目标异常、权重异常）即判为不安全
        has_abnormal_match = torch.any(anomaly_flags["abnormal_match"] > 0.5)
        has_abnormal_target = torch.any(anomaly_flags["infra_abnormal_target"] > 0.5)
        has_weight_outlier = torch.any(anomaly_flags["weight_outlier"] > 0.5)
        is_safe = ~(has_abnormal_match | has_abnormal_target | has_weight_outlier)

        return {
            "association_score": association_score,
            "infra_conf_weight": infra_conf_weight,
            "anomaly_flags": anomaly_flags,
            "is_safe": is_safe
        }
