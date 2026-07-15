import torch
import torch.nn as nn

class TargetEncoder(nn.Module):
    def __init__(
        self,
        in_dim=10,
        embed_dim=64,
        num_classes=10,
        cont_mean=None,   # 形状 [9] 的连续特征均值
        cont_std=None     # 形状 [9] 的连续特征标准差
    ):
        super().__init__()
        # 类别嵌入
        self.cls_embed = nn.Embedding(num_classes, embed_dim // 2)

        # 连续特征归一化参数（可学习或固定）
        if cont_mean is not None and cont_std is not None:
            # 固定统计归一化
            self.register_buffer('cont_mean', torch.tensor(cont_mean, dtype=torch.float32))
            self.register_buffer('cont_std', torch.tensor(cont_std, dtype=torch.float32))
            self.norm_type = 'statistical'
        else:
            # 使用可学习的 LayerNorm（在特征维度上）
            self.cont_norm = nn.LayerNorm(in_dim - 1)  # 9个连续特征
            self.norm_type = 'layernorm'

        # 连续特征编码
        self.cont_fc = nn.Linear(in_dim - 1, embed_dim // 2)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

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
        x: [B, N, 10]
           - 索引 0-7: x, y, z, w, l, h, yaw, score
           - 索引 8: cls（整数）
           - 索引 9: 相对时间差（目标时刻 - 车端时刻），单位秒
        """
        # 1. 提取类别索引（原始值，不参与归一化）
        cls_idx = x[..., 8].long()                     # [B, N]

        # 2. 提取连续特征 (前8维 + 时间差)
        cont_feat = torch.cat([x[..., :8], x[..., 9:10]], dim=-1)  # [B, N, 9]

        # 3. 连续特征归一化
        if self.norm_type == 'statistical':
            # 使用预计算统计量
            mean = self.cont_mean.to(cont_feat.device).view(1, 1, -1)
            std = self.cont_std.to(cont_feat.device).view(1, 1, -1)
            cont_feat = (cont_feat - mean) / (std + 1e-8)
        else:
            # 使用 LayerNorm
            cont_feat = self.cont_norm(cont_feat)

        # 4. 特征编码
        cls_emb = self.cls_embed(cls_idx)              # [B, N, 32]
        cont_out = self.cont_fc(cont_feat)             # [B, N, 32]

        # 5. 拼接融合
        combined = torch.cat([cont_out, cls_emb], dim=-1)  # [B, N, 64]
        return self.fusion(combined)                       # [B, N, 64]
