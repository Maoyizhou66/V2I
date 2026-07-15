import torch
import torch.nn as nn

class TargetEncoder(nn.Module):
    def __init__(
        self,
        in_dim=10,
        embed_dim=64,
        num_classes=10,
        cont_mean=None,   # 可选，形状 [9] 的连续特征均值
        cont_std=None     # 可选，形状 [9] 的连续特征标准差
    ):
        """
        目标特征编码器（改进版）
        Args:
            in_dim: 输入特征维度，固定为10
            embed_dim: 输出嵌入维度
            num_classes: 目标类别数量
            cont_mean: 连续特征（前8维+时间戳）的均值，用于标准化
            cont_std:  连续特征的标准差，用于标准化
        """
        super().__init__()
        # 输入层归一化（缓解尺度差异）
        self.norm_in = nn.LayerNorm(in_dim)

        # 可选统计归一化（在 LayerNorm 之前做，更精细）
        if cont_mean is not None and cont_std is not None:
            self.register_buffer('cont_mean', torch.tensor(cont_mean, dtype=torch.float32))
            self.register_buffer('cont_std', torch.tensor(cont_std, dtype=torch.float32))
        else:
            self.register_buffer('cont_mean', None)
            self.register_buffer('cont_std', None)

        # 类别嵌入
        self.cls_embed = nn.Embedding(num_classes, embed_dim // 2)

        # 连续特征编码：9个连续特征（x,y,z,w,l,h,yaw,score,rel_time_diff）
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
           索引 0-7: x,y,z,w,l,h,yaw,score
           索引 8: cls (整数类别)
           索引 9: 相对时间差 (目标时刻 - 车端时刻)，单位秒
        """
        # 1. 分离类别和连续特征
        cls_idx = x[..., 8].long()                        # [B, N]
        cont_feat = torch.cat([x[..., :8], x[..., 9:10]], dim=-1)  # [B, N, 9]

        # 2. 统计归一化（如果提供了均值和标准差）
        if self.cont_mean is not None and self.cont_std is not None:
            # 扩展维度以匹配 [B, N, 9]
            mean = self.cont_mean.to(cont_feat.device).view(1, 1, -1)
            std = self.cont_std.to(cont_feat.device).view(1, 1, -1)
            cont_feat = (cont_feat - mean) / (std + 1e-8)

        # 3. 输入层归一化（对整体10维做，但可以只对连续特征做，这里保留原设计）
        # 注意：此处对 x 整体做 LayerNorm 可能混合了类别索引，但我们稍后分离，所以可省略或保留。
        # 更好的做法：仅对连续特征归一化，避免类别索引影响。这里改为只对连续特征做 LayerNorm。
        # 为了兼容，我们重新设计：对连续特征单独 LayerNorm，对类别单独 Embedding。
        # 我们采用以下策略：对 cont_feat 做 LayerNorm，再送入 cont_fc。
        cont_feat = nn.functional.layer_norm(cont_feat, cont_feat.shape[-1:])

        # 4. 特征编码
        cls_emb = self.cls_embed(cls_idx)                 # [B, N, 32]
        cont_out = self.cont_fc(cont_feat)                # [B, N, 32]

        # 5. 拼接融合
        combined = torch.cat([cont_out, cls_emb], dim=-1) # [B, N, 64]
        return self.fusion(combined)                      # [B, N, 64]
