"""

"""
from __future__ import annotations
from torch import nn, Tensor
import torch

__all__ = [
    "IGFOut",
]

class IGFOut(nn.Module):
    """
LSTM 版 IGF_out，输入 x 形状 [N, in_dim]（N 为时间长度），输出 [N]。
    """

    def __init__(
        self,
        *,
        in_dim: int = 4, 
        hidden_dim: int = 32,
        n_hidden: int = 1,
        activation: str = "tanh",  # 兼容旧签名，不使用
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=n_hidden,
            batch_first=True,                    # 输入 [B, T, in_dim] B代表批量
            dropout=dropout if n_hidden > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Linear(out_dim, 1), nn.Softplus())        

        # 轻量初始化，避免门饱和；与原 MLP 的小增益 Xavier 一致
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif "bias" in name:
                nn.init.zeros_(p)

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        # ---------- 在这里加入：遗忘门偏置设为正 ----------
        # PyTorch LSTM 的门顺序是 [i, f, g, o]，所以遗忘门切片是 [H:2H]
        with torch.no_grad():
            for l in range(n_hidden):
                # 正向
                b_ih = getattr(self.lstm, f"bias_ih_l{l}")
                b_hh = getattr(self.lstm, f"bias_hh_l{l}")
                b_ih[hidden_dim:2*hidden_dim].fill_(1.0)
                b_hh[hidden_dim:2*hidden_dim].fill_(1.0)
                # 反向（若启用双向）
                if bidirectional:
                    b_ih_r = getattr(self.lstm, f"bias_ih_l{l}_reverse")
                    b_hh_r = getattr(self.lstm, f"bias_hh_l{l}_reverse")
                    b_ih_r[hidden_dim:2*hidden_dim].fill_(1.0)
                    b_hh_r[hidden_dim:2*hidden_dim].fill_(1.0)

    def forward(self, x: Tensor) -> Tensor:
        # 接受 [N, in_dim] 或 [B, N, in_dim]
        if x.ndim == 2:
            x = x.unsqueeze(0)                  # [1, N, in_dim]
        # LSTM 全序列前向
        y, _ = self.lstm(x)                     # [B, N, H]
        y = self.head(y).squeeze(-1)            # [B, N]
        return y.squeeze(0)                     # [N]
