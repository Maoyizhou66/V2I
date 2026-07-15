import torch
import torch.nn as nn

class TargetEncoder(nn.Module):
    def __init__(self, in_dim=10, embed_dim=64, num_classes=10):
        """
        目标特征编码器
        Args:
            in_dim: 输入特征维度，固定为10
            embed_dim: 输出嵌入维度
            num_classes: 目标类别数量（用于Embedding）
        """
        super().__init__()
        # 输入归一化，缓解不同物理量尺度差异
        self.norm_in = nn.LayerNorm(in_dim)

        # 类别索引嵌入 (cls 位于索引8)
        self.cls_embed = nn.Embedding(num_classes, embed_dim // 2)  # 类别嵌入维度为32

        # 连续特征编码：共9个连续特征 (x,y,z,w,l,h,yaw,score,timestamp)
        self.cont_fc = nn.Linear(in_dim - 1, embed_dim // 2)        # 输出32维

        # 融合层：拼接后维度 32+32=64，再升维到embed_dim（此处保持64）
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, x):
        """
        x: [B, N, 10]   (x,y,z,w,l,h,yaw,score,cls,timestamp)
        """
        x = self.norm_in(x)                     # 输入归一化

        cls_idx = x[..., 8].long()              # [B, N]
        cont_feat = torch.cat([x[..., :8], x[..., 9:10]], dim=-1)  # [B, N, 9]

        cls_emb = self.cls_embed(cls_idx)       # [B, N, 32]
        cont_out = self.cont_fc(cont_feat)      # [B, N, 32]

        combined = torch.cat([cont_out, cls_emb], dim=-1)  # [B, N, 64]
        return self.fusion(combined)            # [B, N, 64]
