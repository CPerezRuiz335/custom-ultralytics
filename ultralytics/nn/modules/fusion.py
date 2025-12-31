import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusion(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.cross = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        # self.tau = 2.0  # temperature to prevent g -> 1 collapse

    def forward(self, tpe, vpe, inference=False):
        if tpe is None: return vpe
        if vpe is None: return tpe

        B, N, D = vpe.shape
        tpe1 = tpe[:, :N, :]
        tpe2 = tpe[:, N:, :]

        # cross influence, but vpe anchor stays strong
        attn_out, _ = self.cross(vpe, tpe1, tpe1)

        # gated mix with temperature (slows collapse)
        g = torch.sigmoid(self.gate(torch.cat([vpe, tpe1], dim=-1)))
        fused = g * attn_out + (1 - g) * vpe  # avoid pure vpe dominance

        # preserve norm of vpe (keeps bbox stable)
        vnorm = vpe.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        fused = F.normalize(fused, dim=-1) * vnorm

        if not inference:
            zeros = fused.new_zeros(B, N, D)
            fused = torch.cat([torch.full_like(fused, 0), torch.zeros(tpe2.shape).cuda(), zeros], dim=1)
        
        return fused

class Fusion_attn(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, tpe, vpe, inference=False):
        """
        Apply cross-attention independently per (tpe[i], vpe[i]) pair.
        tpe: [B, N, D]
        vpe: [B, N, D]
        """
        if tpe is None: return vpe
        if vpe is None: return tpe

        B, N, D = vpe.shape
        tpe1 = tpe[:,:N,:]
        tpe2 = tpe[:,N:,:]


        # Run cross-attention *independently* per batch item
        # MultiheadAttention doesn’t support batched attention directly,
        # so we flatten B and N into one dimension
        fused, _ = self.cross_attn(
            query=vpe.reshape(B*N, 1, D),
            key=tpe1.reshape(B*N, 1, D),
            value=tpe1.reshape(B*N, 1, D)
        )

        out = fused.view(B, N, D)

        if not inference:
            zeros = out.new_zeros((out.size(0), N, out.size(2)))
            out = torch.cat((out, torch.zeros(tpe2.shape), zeros), dim=1)

        return out


class Fusion_(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        """
        Initialize Fusion layer with given parameters.
        (Multi-head attention + gated residual)

        Args:
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.lin1 = nn.Linear(embed_dim*2, embed_dim*4)
        self.lin2 = nn.Linear(embed_dim*4, embed_dim*4)
        self.lin3 = nn.Linear(embed_dim*4, embed_dim*4)
        self.lin4 = nn.Linear(embed_dim*4, embed_dim*2)
        self.lin5 = nn.Linear(embed_dim*2, embed_dim)

        self.drop12 = nn.Dropout(0.2)
        self.drop34 = nn.Dropout(0.3)
        self.drop5 = nn.Dropout(0.5)

        self.act = nn.ReLU()

    def forward(self, tpe, vpe, inference=False):
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

        x = torch.concat((tpe_, vpe), dim=-1)
        x = self.drop12(self.act(self.lin1(x)))
        x = self.drop12(self.act(self.lin2(x)))
        x = self.drop34(self.act(self.lin3(x)))
        x = self.drop34(self.act(self.lin4(x)))
        out = self.drop5(self.act(self.lin5(x)))
        
        # out =  vpe + self.gate * fused
        # out = fused

        if not inference:
            pad = tpe.shape[1] + n - out.size(1)

            if pad > 0:
                # zeros with SAME device & dtype as out
                zeros = out.new_zeros((out.size(0), pad, out.size(2)))
                out = torch.cat((out, zeros), dim=1)
            elif pad < 0:
                # trim if longer than 80
                out = out[:, :80, :]

        return out

class Fusion_oldold(nn.Module):
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

    def forward(self, tpe, vpe, inference=False):
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
        # out =  vpe + self.gate * fused
        out = fused

        if not inference:
            pad = tpe.shape[1] + n - out.size(1)

            if pad > 0:
                # zeros with SAME device & dtype as out
                zeros = out.new_zeros((out.size(0), pad, out.size(2)))
                out = torch.cat((out, zeros), dim=1)
            elif pad < 0:
                # trim if longer than 80
                out = out[:, :80, :]

        return out