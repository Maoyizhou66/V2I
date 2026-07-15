
import torch
import torch.nn as nn

# ==================== 改进后的 HardConstraintOutputLayer ====================
class HardConstraintOutputLayer(nn.Module):
    def __init__(
        self,
        score_min=0.0,
        score_max=1.0,
        conf_w_min=0.1,
        conf_w_max=1.0,
        delay_thresh=150,
        iou_thresh=0.1,
        abnormal_score_thresh=0.5,      # 新增：判定异常匹配的分数阈值
        inf_score_thresh=0.2,           # 新增：路端目标分数阈值
        heads_thresh_ratio=0.5          # 新增：异常匹配头部比例阈值（超过该比例才标记）
    ):
        super().__init__()
        self.score_min = score_min
        self.score_max = score_max
        self.conf_w_min = conf_w_min
        self.conf_w_max = conf_w_max
        self.delay_thresh = delay_thresh
        self.iou_thresh = iou_thresh
        self.abnormal_score_thresh = abnormal_score_thresh
        self.inf_score_thresh = inf_score_thresh
        self.heads_thresh_ratio = heads_thresh_ratio   # 用于异常匹配投票

    def forward(self, veh2inf_attn, inf2veh_attn, veh_targets, inf_targets, iou_3d_matrix):
        """
        输入:
            veh2inf_attn: [B, heads, Nv, Ni]
            inf2veh_attn: [B, heads, Ni, Nv]
            veh_targets: [B, Nv, 10]
            inf_targets: [B, Ni, 10]
            iou_3d_matrix: [B, Nv, Ni]
        输出:
            字典
        """
        B, heads, Nv, Ni = veh2inf_attn.shape

        # ---------------------- 1. 关联度分数 ----------------------
        # 对称化融合
        attn_fused = (veh2inf_attn + inf2veh_attn.transpose(2, 3)) * 0.5
        # IoU自适应权重
        iou_weight = torch.where(
            iou_3d_matrix.unsqueeze(1) < self.iou_thresh,
            torch.ones_like(iou_3d_matrix.unsqueeze(1)),
            1.0 - (iou_3d_matrix.unsqueeze(1) - self.iou_thresh) / (1.0 - self.iou_thresh)
        )
        attn_fused = attn_fused * iou_weight
        association_score = torch.clamp(attn_fused, self.score_min, self.score_max)

        # ---------------------- 2. 路端置信度权重 ----------------------
        # 聚合车端对路端的注意力：先平均头部，再在车端维度聚合 (使用均值+最大值组合，更鲁棒)
        veh2inf_avg = veh2inf_attn.mean(dim=1)  # [B, Nv, Ni]
        # 沿车端维度取均值和最大值
        attn_mean = veh2inf_avg.mean(dim=1)     # [B, Ni]
        attn_max = veh2inf_avg.max(dim=1)[0]    # [B, Ni]
        # 组合：均值 + 0.5*最大值，兼顾整体关注度和峰值
        inf_attn_agg = attn_mean + 0.5 * attn_max   # [B, Ni]

        # 路端目标属性
        inf_score = inf_targets[..., 7]      # [B, Ni]
        inf_delay = inf_targets[..., 9]      # [B, Ni]
        delay_penalty = torch.exp(-inf_delay / self.delay_thresh)  # [B, Ni]

        # 原始权重（未clamp）
        conf_weight_raw = inf_attn_agg * inf_score * delay_penalty   # [B, Ni]

        # **修正：先检测异常，再clamp**
        # 权重异常标记（基于原始值）
        weight_outlier = ((conf_weight_raw < self.conf_w_min) | (conf_weight_raw > self.conf_w_max)).float()

        # 再clamp到合法范围
        infra_conf_weight = torch.clamp(conf_weight_raw, self.conf_w_min, self.conf_w_max)

        # ---------------------- 3. 异常标记 ----------------------
        anomaly_flags = {}

        # 异常匹配：每个头独立检测，然后投票（超过阈值比例的头认为异常）
        # 检测每个头：高关联度且低IoU
        abnormal_match_per_head = ((association_score > self.abnormal_score_thresh) &
                                   (iou_3d_matrix.unsqueeze(1) < self.iou_thresh)).float()  # [B, heads, Nv, Ni]
        # 统计异常头数
        abnormal_head_count = abnormal_match_per_head.sum(dim=1)   # [B, Nv, Ni] 值为0~heads
        # 投票：若异常头数超过 heads * heads_thresh_ratio，则标记为异常匹配
        abnormal_match_vote = (abnormal_head_count > (self.heads_thresh_ratio * heads)).float()
        anomaly_flags["abnormal_match"] = abnormal_match_vote      # [B, Nv, Ni]

        # 路端异常目标（延迟过高或分数过低）
        infra_abnormal = ((inf_delay > self.delay_thresh) | (inf_score < self.inf_score_thresh)).float()
        anomaly_flags["infra_abnormal_target"] = infra_abnormal    # [B, Ni]

        # 权重异常（已计算）
        anomaly_flags["weight_outlier"] = weight_outlier           # [B, Ni]

        # ---------------------- 4. 安全判断 ----------------------
        # 检查是否存在任何异常（在有效维度上）
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
