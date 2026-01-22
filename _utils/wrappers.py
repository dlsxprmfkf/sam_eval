import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MLPBlockWrapper(nn.Module):
    """
    Wrapper for Dynamic Structured Pruning on a ViT-style MLP block (for sparsity_type='sp').
    This prunes entire neurons based on input activations and weights during the forward pass.
    """
    def __init__(self,
                 mlp_mod: nn.Module,
                 *,
                 layer_id: int,
                 expert_id: int = 0,
                 act_mode: str = "default",
                 rebuild_freq: int = 1,
                 prune_mlp_ratio: float = 0.0,
                 protect_frac: float = 0.01,
                 **kwargs,
                 ):
        super().__init__()
        self.mlp = mlp_mod
        self.layer_id = int(layer_id)
        self.expert_id = int(expert_id)
        self.act_mode = act_mode
        self.prune_mlp_ratio = float(prune_mlp_ratio)
        self.protect_frac = float(protect_frac)

        # Assign linear layers based on SAM's MLP structure
        self.lin1 = self.mlp.lin1
        self.lin2 = self.mlp.lin2
        
        # Store intermediate and hidden dimensions
        self.I = self.lin1.weight.shape[0]
        self.H = self.lin1.weight.shape[1]

        self.forward_count = 0
        self.initialized = False
        self.rebuild_freq = int(rebuild_freq)

    def get_scaler(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # Reshape input tensor to [TotalTokens, HiddenDim]
        if x.dim() == 4: # [B, H, W, C] for SAM
            B, H, W, C = x.shape
            inp = x.reshape(B * H * W, C)
        elif x.dim() == 3: # [B, N, H]
            B, L, H = x.shape
            inp = x.reshape(B * L, H)
        else:
            inp = x
        
        # Calculate RMS scaler
        inp = inp.to(torch.float32)
        mean2 = (inp * inp).mean(dim=0)
        scaler = torch.sqrt(mean2 + eps)
        return scaler

    @torch.no_grad()
    def _build_keep_mask(self, x: torch.Tensor, ratio: float) -> torch.Tensor:
        # Calculate importance scores for each neuron
        I, H = self.I, self.H
        g_hidden = self.get_scaler(x).to(torch.float32).reshape(1, H)
        
        W_lin1 = self.lin1.weight.detach().abs().to(torch.float32)
        score  = torch.linalg.vector_norm(W_lin1 * g_hidden, ord=2, dim=1)

        # Determine which neurons to keep based on scores
        dna = int(self.protect_frac * I)
        _, idx_sorted = torch.sort(score, dim=0, descending=False)
        idx_protect = idx_sorted[:dna]
        idx_rest    = idx_sorted[dna:]
        keep_ct = max(1, int(round((1.0 - ratio) * I)))
        prune_ct = max(0, min(idx_rest.numel(), I - keep_ct))
        
        idx_kept = torch.cat([idx_protect, idx_rest[prune_ct:]], dim=0)
        idx_kept, _ = torch.sort(idx_kept)
        return idx_kept

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        ratio = self.prune_mlp_ratio
        rebuild = ((not self.initialized) or (self.rebuild_freq == 1) or (self.forward_count % self.rebuild_freq == 1)) and ratio > 0.0

        if not rebuild:
            return self.mlp(x) # If not pruning, run original MLP
        
        self.initialized = True
        with torch.no_grad():
            idx_kept = self._build_keep_mask(x, ratio)

        # Slice weights in real-time based on the keep mask
        W_lin1_k = torch.index_select(self.lin1.weight, 0, idx_kept).contiguous()
        b_lin1_k = torch.index_select(self.lin1.bias, 0, idx_kept).contiguous() if self.lin1.bias is not None else None
        
        W_lin2_k = torch.index_select(self.lin2.weight, 1, idx_kept).contiguous()
        b_lin2   = self.lin2.bias

        # Perform forward pass with smaller, sliced weights
        y1 = F.linear(x, W_lin1_k, b_lin1_k)
        h = self.mlp.act(y1)
        out = F.linear(h, W_lin2_k, b_lin2)
        
        return out

class LinearWrapperUnstructured(nn.Module):
    """
    Wrapper for Dynamic Unstructured Pruning.
    This zeros out individual weights based on their magnitude AND input activations.
    The mask is rebuilt periodically.
    """
    def __init__(self,
                 linear_mod: nn.Module,
                 *,
                 prune_ratio: float = 0.0,
                 rebuild_freq: int = 1,
                 act_mode: str = 'rms',
                 **kwargs,
                 ):
        super().__init__()
        self.linear = linear_mod
        self.prune_ratio = float(prune_ratio)
        self.rebuild_freq = int(rebuild_freq)
        self.act_mode = act_mode
        self.mask = None
        self.initialized = False
        self.forward_count = 0

    def get_scaler(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Calculates a scaler based on input activations."""
        if x.dim() == 4:
            B, H, W, C = x.shape
            inp = x.reshape(B * H * W, C)
        elif x.dim() == 3:
            B, L, H = x.shape
            inp = x.reshape(B * L, H)
        else:
            inp = x

        inp = inp.to(torch.float32)
        mean2 = (inp * inp).mean(dim=0)
        scaler = torch.sqrt(mean2 + eps)
        return scaler

    @torch.no_grad()
    def _build_mask(self, x: torch.Tensor):
        """Builds mask based on weight magnitude and input activation."""
        if self.prune_ratio > 0:
            scaler = self.get_scaler(x).to(torch.float32)
            W = self.linear.weight.detach().abs()
            score = W * scaler.reshape(1, -1)

            flat_score = score.flatten()
            k = int(self.prune_ratio * flat_score.numel())
            
            if k < 1:
                self.mask = torch.ones_like(self.linear.weight)
                return

            threshold = torch.kthvalue(flat_score, k).values
            self.mask = (score > threshold).to(self.linear.weight.dtype)
        else:
            self.mask = torch.ones_like(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        
        rebuild = (not self.initialized) or \
                  (self.rebuild_freq == 1) or \
                  (self.forward_count % self.rebuild_freq == 1)

        if rebuild and self.prune_ratio > 0.0:
            self._build_mask(x)
            self.initialized = True
        
        if self.prune_ratio == 0.0:
            return F.linear(x, self.linear.weight, self.linear.bias)

        if self.mask is None:
            self._build_mask(x)
            self.initialized = True

        pruned_weight = self.linear.weight * self.mask
        return F.linear(x, pruned_weight, self.linear.bias)
