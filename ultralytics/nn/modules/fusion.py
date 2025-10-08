import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        """
        Initialize Fusion layer with given parameters.
        (Multi-head attention + gated residual)

        Args:
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, tpe, vpe):
        """
        Apply multi-head attention and gated residual to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        # print(f"(Fusion forward) tpe={tpe}, vpe={vpe}")
        if tpe is None: return vpe
        if vpe is None: return tpe

        n = vpe.shape[1]
        tpe_ = tpe[:,:n,:]
        fused, _ = self.cross_attn(query=vpe, key=tpe_, value=tpe_)
        out =  vpe + self.gate * fused
        
        pad = tpe.shape[1] + n - out.size(1)
        if pad > 0:
            # zeros with SAME device & dtype as out
            zeros = out.new_zeros((out.size(0), pad, out.size(2)))
            out = torch.cat((out, zeros), dim=1)
        elif pad < 0:
            # trim if longer than 80
            out = out[:, :80, :]

        return out