# This version on Nov. 2025 is a bspline surface VAE, dedicated to support Bspline Surfaces with variable knots, mults and poles
# The input is the uv knots list, mults list, pole lists, and u_degree, v_degree, u_periodic and v_periodic.
# The output recovers all the above and should reconstruct the exact surface.
# This V3 version use hybrid grid query generator to generate the poles queries.
# However it's not useful for the first test... May try other runs later.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from icecream import ic

ic.disable()




def sinusoidal_pe(L, d, device):
    pos = torch.arange(L, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float, device=device) *
                         (-math.log(10000.0) / d))
    pe = torch.zeros(L, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe

class Sinusoidal2DPositionalEncoding(nn.Module):
    """
    2D sinusoidal positional encoding for grid-structured data.
    Supports arbitrary height (H) and width (W).
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D PE"
        self.half_dim = embed_dim // 2  # split for i and j

    def forward(self, H: int, W: int, device: torch.device):
        """
        Returns: [H, W, embed_dim]
        """
        # Compute position indices
        i_pos = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1)  # [H, 1]
        j_pos = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0)  # [1, W]

        # Frequency scaling (like original Transformer)
        div_term = torch.exp(
            torch.arange(0, self.half_dim, 2, dtype=torch.float32, device=device) *
            (-math.log(10000.0) / self.half_dim)
        )  # [d/4]

        # Encode i (row)
        pe_i = torch.zeros(H, self.half_dim, device=device)
        pe_i[:, 0::2] = torch.sin(i_pos * div_term)   # [H, d/4]
        pe_i[:, 1::2] = torch.cos(i_pos * div_term)

        # Encode j (col)
        pe_j = torch.zeros(W, self.half_dim, device=device)
        pe_j[:, 0::2] = torch.sin(j_pos * div_term)   # [W, d/4]
        pe_j[:, 1::2] = torch.cos(j_pos * div_term)

        # Broadcast and concatenate
        pe_i = pe_i.unsqueeze(1)      # [H, 1, d/2]
        pe_j = pe_j.unsqueeze(0)      # [1, W, d/2]
        pe = torch.cat([pe_i.expand(-1, W, -1), 
                        pe_j.expand(H, -1, -1)], dim=-1)  # [H, W, d]

        return pe

def normalized_2d_pe(H: int, W: int, embed_dim: int, device: torch.device):
    """
    2D positional encoding based on normalized coordinates (i/H, j/W) ∈ [0,1]
    Returns: [H, W, embed_dim]
    """
    if H == 0 or W == 0:
        return torch.zeros(0, 0, embed_dim, device=device)
    
    # Normalized coordinates
    i_norm = torch.arange(H, device=device).float() / max(H - 1, 1)  # [H]
    j_norm = torch.arange(W, device=device).float() / max(W - 1, 1)  # [W]
    
    # Prepare frequency scaling
    half_dim = embed_dim // 2
    div_term = torch.exp(
        torch.arange(0, half_dim, 2, dtype=torch.float32, device=device) *
        (-math.log(10000.0) / half_dim)
    )  # [d/4]
    
    # Encode i_norm
    pe_i = torch.zeros(H, half_dim, device=device)
    pe_i[:, 0::2] = torch.sin(i_norm.unsqueeze(1) * div_term)
    pe_i[:, 1::2] = torch.cos(i_norm.unsqueeze(1) * div_term)
    
    # Encode j_norm
    pe_j = torch.zeros(W, half_dim, device=device)
    pe_j[:, 0::2] = torch.sin(j_norm.unsqueeze(1) * div_term)
    pe_j[:, 1::2] = torch.cos(j_norm.unsqueeze(1) * div_term)
    
    # Combine
    pe_i = pe_i.unsqueeze(1).expand(-1, W, -1)  # [H, W, d/2]
    pe_j = pe_j.unsqueeze(0).expand(H, -1, -1)  # [H, W, d/2]
    pe = torch.cat([pe_i, pe_j], dim=-1)        # [H, W, d]
    return pe


class LearnedRelativePE1D(nn.Module):
    """
    1D Relative Positional Encoding using normalized coordinates t = i / (M - 1)
    - Works for any M >= 1
    - Output shape: (M, embed_dim)
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, M: int, device: torch.device = None) -> torch.Tensor:
        """
        Args:
            M (int): sequence length (number of unique knots)
            device (torch.device): device to place the tensor on

        Returns:
            pe (torch.Tensor): (M, embed_dim)
        """
        if M == 1:
            # Edge case: single knot → t = 0
            t = torch.zeros(1, 1, device=device)  # (1, 1)
        else:
            indices = torch.arange(M.item(), dtype=torch.float32, device=device)  # (M,)
            t = indices / (M - 1)  # normalize to [0, 1]
            t = t.unsqueeze(-1)    # (M, 1)

        pe = self.mlp(t)  # (M, embed_dim)
        return pe

class LearnedRelativePE2D(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        # Input: (u, v) ∈ [0,1]^2 → embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, H: int, W: int, device: torch.device = None) -> torch.Tensor:
        """
        Returns:
            pe: (H, W, embed_dim)
        """
        if H == 0 or W == 0:
            return torch.zeros(H, W, self.embed_dim, device=device)
        
        # Normalized coordinates: (H,) and (W,)
        u = torch.arange(H.item(), device=device, dtype=torch.float32) / max(H - 1, 1)
        v = torch.arange(W.item(), device=device, dtype=torch.float32) / max(W - 1, 1)
        
        # Meshgrid: (H, W, 2)
        u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')  # PyTorch >=1.10
        coords = torch.stack([u_grid, v_grid], dim=-1)        # (H, W, 2)
        
        # Encode
        pe = self.mlp(coords)  # (H, W, embed_dim)
        return pe


class PoleEmbedder(nn.Module):
    """
    Embedder for B-spline/NURBS poles (homogeneous coordinates [x, y, z, w]).
    Uses 2D sinusoidal positional encoding + geometric feature projection.
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim should be divisible by 4 for clean 2D PE"
        self.embed_dim = embed_dim
        # self.geom_proj = nn.Linear(4, embed_dim)  # [x,y,z,w] -> embed_dim
        self.geom_proj = nn.Sequential(
            nn.Linear(4, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )  # [x,y,z,w] -> embed_dim

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, poles: torch.Tensor):
        """
        Args:
            poles: [B, H, W, 4] — homogeneous control points (x, y, z, w)
        Returns:
            tokens: [B, H*W, embed_dim]
        """
        B, H, W, _ = poles.shape
        device = poles.device

        # 1. Project geometric features
        geom_feat = self.geom_proj(poles)  # [B, H, W, embed_dim]

        # 2. Generate 2D positional encoding

        # 3. Add and normalize
        tokens = self.norm(geom_feat)  # [B, H, W, embed_dim]


        return tokens





class HybridGridQueryGenerator(nn.Module):
    """
    Hybrid Grid Query Generator for variable-size pole grids.
    - Base: 8x8 learnable canonical grid (structure prior)
    - Refinement: MLP on normalized (u,v) coordinates (detail & adaptivity)
    - Output: [B, H, W, D] queries for pole decoding
    """
    def __init__(self, embed_dim: int, canonical_size: int = 8, refine_hidden: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.canonical_size = canonical_size
        
        # 1. Canonical grid: 8x8 learnable base structure
        self.canonical_queries = nn.Parameter(
            torch.randn(canonical_size, canonical_size, embed_dim)
        )
        # Xavier-like init for stability
        nn.init.normal_(self.canonical_queries, std=0.02)
        
        # 2. Refinement MLP: f(u, v) → Δq, (u,v) ∈ [0, 1]^2
        self.refine_mlp = nn.Sequential(
            nn.Linear(2, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, embed_dim)
        )
        # Initialize refinement to zero mean (start from canonical)
        nn.init.zeros_(self.refine_mlp[-1].weight)
        nn.init.zeros_(self.refine_mlp[-1].bias)
    
    def _generate_normalized_coords(self, H: int, W: int, device: torch.device):
        """
        Generate normalized (u, v) coordinates in [0, 1].
        Handles edge cases:
          - H=1 or W=1: returns 0.0 (center)
          - H>1, W>1: linspace(0, 1, H/W)
        Returns:
            coords: [H, W, 2]
        """
        if H == 1 and W == 1:
            u = torch.tensor([0.0], device=device)
            v = torch.tensor([0.0], device=device)
        elif H == 1:
            u = torch.tensor([0.0], device=device)
            v = torch.linspace(0.0, 1.0, W, device=device)
        elif W == 1:
            u = torch.linspace(0.0, 1.0, H, device=device)
            v = torch.tensor([0.0], device=device)
        else:
            u = torch.linspace(0.0, 1.0, H, device=device)
            v = torch.linspace(0.0, 1.0, W, device=device)
        
        u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')  # [H, W]
        coords = torch.stack([u_grid, v_grid], dim=-1)        # [H, W, 2]
        return coords

    def forward(self, 
                batch_size: int,
                target_h_list: torch.Tensor,   # [B], e.g., tensor([4, 17, 64])
                target_w_list: torch.Tensor,   # [B], e.g., tensor([2, 9, 32])
                max_h: int,
                max_w: int,
                device: torch.device):
        """
        Generate batch of queries with dynamic sizes.
        
        Args:
            batch_size: int
            target_h_list: [B] int tensor of target heights
            target_w_list: [B] int tensor of target widths
            max_h, max_w: global max sizes (e.g., 64, 32)
            device: torch.device
            
        Returns:
            queries_padded: [B, max_h, max_w, embed_dim]
            mask: [B, max_h, max_w] (True where valid)
        """
        B = batch_size
        D = self.embed_dim
        queries_list = []
        mask_list = []
        
        # Expand canonical to batch: [B, 8, 8, D]
        canonical_batch = self.canonical_queries.unsqueeze(0).expand(B, -1, -1, -1)
        
        for i in range(B):
            H = int(target_h_list[i].item())
            W = int(target_w_list[i].item())
            
            # Guard: clamp to [1, max]
            H = max(1, min(H, max_h))
            W = max(1, min(W, max_w))
            
            # Step 1: Resize canonical grid to (H, W) via bilinear interpolation
            # Input: [1, D, 8, 8] → Output: [1, D, H, W]
            canonical_resized = F.interpolate(
                canonical_batch[i:i+1].permute(0, 3, 1, 2),  # [1, D, 8, 8]
                size=(H, W),
                mode='bilinear',
                align_corners=True  # Critical for small grids (e.g., 2x2)
            ).permute(0, 2, 3, 1)  # [1, H, W, D]
            
            # Step 2: Generate refinement Δq using (u,v) coordinates
            coords = self._generate_normalized_coords(H, W, device)  # [H, W, 2]
            delta = self.refine_mlp(coords)  # [H, W, D]
            delta = delta.unsqueeze(0)       # [1, H, W, D]
            
            # Step 3: Combine: q = canonical + delta
            queries_hw = canonical_resized + delta  # [1, H, W, D]
            
            # Step 4: Pad to (max_h, max_w)
            pad_right = max_w - W
            pad_bottom = max_h - H
            queries_padded = F.pad(
                queries_hw,
                (0, 0, 0, pad_right, 0, pad_bottom),
                mode='constant',
                value=0.0
            )  # [1, max_h, max_w, D]
            
            # Step 5: Create mask
            mask = torch.zeros(max_h, max_w, dtype=torch.bool, device=device)
            mask[:H, :W] = True
            
            queries_list.append(queries_padded)
            mask_list.append(mask.unsqueeze(0))
        
        queries_batch = torch.cat(queries_list, dim=0)  # [B, max_h, max_w, D]
        mask_batch = torch.cat(mask_list, dim=0)        # [B, max_h, max_w]
        
        return queries_batch, mask_batch





class CrossAttentionFuser(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Three independent cross-attention layers
        self.cross_attn_u = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_v = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_p = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Optional: LayerNorm for stability
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, Q, U, V, P, meta, U_padding_mask, V_padding_mask, P_padding_mask):
        """
        Args:
            Q: [B, N_q, D] — latent queries
            K_u, V_u: [B, L_u, D] — u-knots
            K_v, V_v: [B, L_v, D] — v-knots
            K_p, V_p: [B, N_p, D] — poles
        Returns:
            Z: [B, N_q, D] — fused latent representation
        """

        # First concat the meta information
        B = U.shape[0]
        U = torch.cat([meta, U], dim=1)
        V = torch.cat([meta, V], dim=1)
        P = torch.cat([meta, P], dim=1)

        U_padding_mask = torch.cat([torch.zeros(B, 1, device=U.device, dtype=torch.bool), U_padding_mask], dim=1)
        V_padding_mask = torch.cat([torch.zeros(B, 1, device=V.device, dtype=torch.bool), V_padding_mask], dim=1)
        P_padding_mask = torch.cat([torch.zeros(B, 1, device=P.device, dtype=torch.bool), P_padding_mask], dim=1)
        K_u = U
        V_u = U
        K_v = V
        V_v = V
        K_p = P
        V_p = P
        # Cross-attention with u-knots
        attn_u, _ = self.cross_attn_u(Q, K_u, V_u, key_padding_mask=U_padding_mask)  # [B, N_q, D]
        
        # Cross-attention with v-knots
        attn_v, _ = self.cross_attn_v(Q, K_v, V_v, key_padding_mask=V_padding_mask)  # [B, N_q, D]
        
        # Cross-attention with poles
        attn_p, _ = self.cross_attn_p(Q, K_p, V_p, key_padding_mask=P_padding_mask)  # [B, N_q, D]
        
        # Fuse: simple sum (you can also use learned weights)
        Z = attn_u + attn_v + attn_p  # [B, N_q, D]
        
        # Optional: LayerNorm
        Z = self.norm(Z)
        
        return Z


class NonlinearMLP(nn.Module):
    def __init__(self, input_dim, model_dim: int = 64, output_dim: int = None, num_heads: int = 4):
        # Dummy heads for num_heads
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.GELU(),          # 或 nn.ReLU()
            nn.Linear(model_dim, output_dim)
        )
    
    def forward(self, x, padding_mask=None):
        return self.net(x)


class NonlinearMLPwithAttn(nn.Module):
    def __init__(self, input_dim: int, model_dim: int = 64, output_dim: int = None, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        if output_dim is None:
            output_dim = input_dim
        self.self_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim)
        )
        self.output_proj = nn.Linear(model_dim, output_dim)

    def forward(self, x, padding_mask):
        """
        x: [B, L, input_dim]
        mask: [B, L]
        returns K, V: [B, L, model_dim]
        """
        x = self.input_proj(x)  # [B, L, model_dim]
        
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=padding_mask)
        x = self.norm1(x + attn_out)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        x = self.output_proj(x)
        return x # K = V = x



class BSplineVAE(nn.Module):
    def __init__(self, 
                 max_degree=3,  # Support up to 3-degree bsplines
                 embd_dim=64,       # 潜空间维度
                 num_query=32,   # Number of queries
                 mults_dim=16,   # Dimension of mults embedding
                 max_num_u_knots=64,
                 max_num_v_knots=32,
                 max_num_u_poles=64,
                 max_num_v_poles=32,
                 ):

        super().__init__()
        
        self.max_degree = max_degree
        self.max_num_u_knots = max_num_u_knots
        self.max_num_v_knots = max_num_v_knots
        self.max_num_u_poles = max_num_u_poles
        self.max_num_v_poles = max_num_v_poles
        self.embd_dim = embd_dim

        self.u_mults_embed = nn.Embedding(num_embeddings=max_degree+1, embedding_dim=mults_dim)
        self.v_mults_embed = nn.Embedding(num_embeddings=max_degree+1, embedding_dim=mults_dim)
        


        self.deg_embed = nn.Embedding(num_embeddings=max_degree, embedding_dim=embd_dim)
        self.periodic_embed = nn.Embedding(num_embeddings=2, embedding_dim=embd_dim)

        self.latent_queries = nn.Parameter(torch.randn(num_query, embd_dim))
        nn.init.normal_(self.latent_queries, std=1.0 / math.sqrt(embd_dim))



        # Encoder 
        self.u_knots_proj = nn.Linear(1, embd_dim)
        self.v_knots_proj = nn.Linear(1, embd_dim)

 
        # Remove Attn
        self.U_encoder = NonlinearMLPwithAttn(input_dim=mults_dim + embd_dim, model_dim=embd_dim, output_dim=embd_dim, num_heads=4)
        self.V_encoder = NonlinearMLPwithAttn(input_dim=mults_dim + embd_dim, model_dim=embd_dim, output_dim=embd_dim, num_heads=4)

        self.u_knots_pe = LearnedRelativePE1D(embed_dim=embd_dim)
        self.v_knots_pe = LearnedRelativePE1D(embed_dim=embd_dim)

        self.poles_embedder = PoleEmbedder(embed_dim=embd_dim)

        self.poles_pe = LearnedRelativePE2D(embed_dim=embd_dim)

        self.poles_encoder = NonlinearMLPwithAttn(input_dim=embd_dim, model_dim=embd_dim, num_heads=4)

        self.fuser = CrossAttentionFuser(embed_dim=embd_dim, num_heads=4)

        self.meta_proj = nn.Linear(embd_dim * 4, embd_dim)

        
        # 输出潜空间参数
        self.fc_mu = nn.Linear(embd_dim, embd_dim)
        self.fc_logvar = nn.Linear(embd_dim, embd_dim)
        
        # Decoder
        # TODO

        
        self.deg_head_u = nn.Linear(embd_dim, max_degree)  # max_degree=3 → classes: deg1,deg2,deg3
        self.deg_head_v = nn.Linear(embd_dim, max_degree)

        self.peri_head_u = nn.Linear(embd_dim, 1) # 0 or 1
        self.peri_head_v = nn.Linear(embd_dim, 1)

        self.knots_num_head_u = nn.Linear(embd_dim, max_num_u_knots)
        self.knots_num_head_v = nn.Linear(embd_dim, max_num_v_knots)
        
        self.token_proj_u = NonlinearMLP(embd_dim, embd_dim, max_num_u_knots * embd_dim)
        self.token_proj_v = NonlinearMLP(embd_dim, embd_dim, max_num_v_knots * embd_dim)

        # Remove attn
        self.knots_head_u = NonlinearMLPwithAttn(embd_dim, embd_dim, 1)
        self.knots_head_v = NonlinearMLPwithAttn(embd_dim, embd_dim, 1)
        self.mults_head_u = NonlinearMLPwithAttn(embd_dim, embd_dim, max_degree + 1)
        self.mults_head_v = NonlinearMLPwithAttn(embd_dim, embd_dim, max_degree + 1)
        self.softplus = nn.Softplus()

        # self.token_proj_poles = NonlinearMLP(embd_dim, embd_dim, embd_dim)
        self.poles_query_generator = HybridGridQueryGenerator(
            embed_dim=embd_dim,
            canonical_size=8,
            refine_hidden=64
        )
        self.poles_head = NonlinearMLPwithAttn(embd_dim, embd_dim, 4)





    # 
    def encode(self, u_knots, u_mults, v_knots, v_mults, poles, u_degree, v_degree, u_periodic, v_periodic, num_knots_u, num_knots_v, num_poles_u, num_poles_v):

        B = u_knots.shape[0]
        # Minus 1 to start from 0
        # u_degree = u_degree - 1
        # v_degree = v_degree - 1
        # u_mults = u_mults - 1
        # v_mults = v_mults - 1

        # Get the embedding of degree and periodic
        u_deg_embd = self.deg_embed(u_degree) # (B, 1, embd_dim)
        v_deg_embd = self.deg_embed(v_degree) # (B, 1, embd_dim)
        u_periodic_embd = self.periodic_embed(u_periodic) # (B, 1, embd_dim)
        v_periodic_embd = self.periodic_embed(v_periodic) # (B, 1, embd_dim)

        # Get the embedding of knots and mults
        u_knots_embd = self.u_knots_proj(u_knots.unsqueeze(-1)) # (B, L_u, embd_dim)
        v_knots_embd = self.v_knots_proj(v_knots.unsqueeze(-1)) # (B, L_v, embd_dim)
        u_mults_embd = self.u_mults_embed(u_mults) # (B, L_u, mults_dim)
        v_mults_embd = self.v_mults_embed(v_mults) # (B, L_v, mults_dim)

        # Add positional encoding to knots and mults
        # u_knots_pe = sinusoidal_pe(u_knots.shape[1], self.embd_dim, u_knots.device) # (L_u, embd_dim)
        # v_knots_pe = sinusoidal_pe(v_knots.shape[1], self.embd_dim, v_knots.device) # (L_v, embd_dim)
        u_knots_pe = torch.zeros(B, self.max_num_u_knots, self.embd_dim, device=u_knots.device)
        v_knots_pe = torch.zeros(B, self.max_num_v_knots, self.embd_dim, device=v_knots.device)
        u_knots_pe_mask = torch.zeros(B, self.max_num_u_knots, device=u_knots.device, dtype=torch.bool)
        v_knots_pe_mask = torch.zeros(B, self.max_num_v_knots, device=v_knots.device, dtype=torch.bool)


        for i in range(B):
            _u_knots_pe_i = self.u_knots_pe(num_knots_u[i], device=num_knots_u.device) # (1, embd_dim)
            _v_knots_pe_i = self.v_knots_pe(num_knots_v[i], device=num_knots_v.device) # (1, embd_dim)
            u_knots_pe[i, :num_knots_u[i], :] = _u_knots_pe_i
            v_knots_pe[i, :num_knots_v[i], :] = _v_knots_pe_i
            u_knots_pe_mask[i, :num_knots_u[i]] = 1
            v_knots_pe_mask[i, :num_knots_v[i]] = 1


        u_vector = torch.cat([u_knots_embd + u_knots_pe, u_mults_embd], dim=-1) # (B, L_u, embd_dim + mults_dim)
        v_vector = torch.cat([v_knots_embd + v_knots_pe, v_mults_embd], dim=-1) # (B, L_v, embd_dim + mults_dim)

        u_vector = u_vector * u_knots_pe_mask.unsqueeze(-1)
        v_vector = v_vector * v_knots_pe_mask.unsqueeze(-1)

        U_padding_mask = ~u_knots_pe_mask
        V_padding_mask = ~v_knots_pe_mask

        # Encode knots and mults
        u_encoder_output = self.U_encoder(u_vector, padding_mask = U_padding_mask) # (B, L_u, embd_dim)
        v_encoder_output = self.V_encoder(v_vector, padding_mask = V_padding_mask) # (B, L_v, embd_dim)

        # Encode poles

        # poles_pe = self.poles_pe(poles.shape[0], poles.shape[1], poles.device).unsqueeze(0) # (1, H, W, embd_dim)
        poles_pe = torch.zeros(B, self.max_num_u_poles, self.max_num_v_poles, self.embd_dim, device=v_encoder_output.device)
        poles_mask = torch.zeros(B, self.max_num_u_poles, self.max_num_v_poles, device=v_encoder_output.device, dtype=torch.bool)
        for i in range(B):
            _poles_pe_i = self.poles_pe(num_poles_u[i], num_poles_v[i], poles.device) # (h, w, embd_dim)
            poles_pe[i, :num_poles_u[i], :num_poles_v[i], :] = _poles_pe_i
            poles_mask[i, :num_poles_u[i], :num_poles_v[i]] = 1


        poles = poles * poles_mask.unsqueeze(-1)
        poles_embd = self.poles_embedder(poles) # (B, H, W, embd_dim)
        poles_embd = poles_pe + poles_embd
        poles_embd = einops.rearrange(poles_embd, 'b h w d -> b (h w) d')
        poles_padding_mask = ~poles_mask
        poles_padding_mask = einops.rearrange(poles_padding_mask, 'b h w -> b (h w)')

        P_padding_mask = poles_padding_mask


        poles_encoder_output = self.poles_encoder(poles_embd, padding_mask = poles_padding_mask) # (B, N_p, embd_dim)


        # Collect the U, V Poles padding mask

        # Fuse the encoded information
        query_embd = self.latent_queries.unsqueeze(0).expand(B, -1, -1) # (B, num_query, embd_dim)
        # query_embd = einops.rearrange(query_embd, 'n d -> 1 n d') # (1, num_query, embd_dim)
        # query_embd = query_embd.expand(u_encoder_output.shape[0], -1, -1) # (B, num_query, embd_dim)

        meta_embd = torch.cat([u_deg_embd, v_deg_embd, u_periodic_embd, v_periodic_embd], dim=-1)
        meta_embd = self.meta_proj(meta_embd) # (B, 1, embd_dim)

        # meta_embd_empty = torch.zeros_like(meta_embd)
        fuser_output = self.fuser(query_embd, u_encoder_output, v_encoder_output, poles_encoder_output, meta_embd, U_padding_mask, V_padding_mask, P_padding_mask) # (B, num_query, embd_dim)
        
        # Try move the injection after fuser.
        # fuser_output = self.fuser(query_embd, u_encoder_output, v_encoder_output, poles_encoder_output, meta_embd, U_padding_mask, V_padding_mask, P_padding_mask) # (B, num_query, embd_dim)
        
        # fuser_output = fuser_output + meta_embd
        fuser_output = fuser_output
        fuser_output = fuser_output.mean(dim=1) # (B, embd_dim)

        # Inject the information of degree and periodic
        
        # fuser_output = fuser_output + meta_embd.squeeze(1)

        # Project the fused information to the latent space
        mu = self.fc_mu(fuser_output) # (B, embd_dim)
        logvar = self.fc_logvar(fuser_output) # (B, embd_dim)
       
        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    


    def decode_knots_mults(self, z, num_knots_u, num_knots_v):
        B = z.shape[0]
        device = z.device

        knots_padding_mask_u = torch.zeros(B, self.max_num_u_knots, device=device, dtype=torch.bool)
        knots_padding_mask_v = torch.zeros(B, self.max_num_v_knots, device=device, dtype=torch.bool)
        for i in range(B):
            knots_padding_mask_u[i, :num_knots_u[i]] = 1
            knots_padding_mask_v[i, :num_knots_v[i]] = 1
        knots_padding_mask_u = ~knots_padding_mask_u
        knots_padding_mask_v = ~knots_padding_mask_v

        knots_tokens_u = self.token_proj_u(z) # (B, max_num_u_knots * embd_dim)
        knots_tokens_v = self.token_proj_v(z) # (B, max_num_v_knots * embd_dim)
        knots_tokens_u = knots_tokens_u.view(B, -1, self.embd_dim) # (B, max_num_u_knots, embd_dim)
        knots_tokens_v = knots_tokens_v.view(B, -1, self.embd_dim) # (B, max_num_v_knots, embd_dim)

        knots_pe_u = torch.zeros(B, self.max_num_u_knots, self.embd_dim, device=device) # (B, max_num_u_knots, embd_dim)
        knots_pe_v = torch.zeros(B, self.max_num_v_knots, self.embd_dim, device=device) # (B, max_num_v_knots, embd_dim)
        for i in range(B):
            _knots_pe_u_i = self.u_knots_pe(num_knots_u[i], device=num_knots_u.device).unsqueeze(0) # (1, num_knots_u, embd_dim)
            _knots_pe_v_i = self.v_knots_pe(num_knots_v[i], device=num_knots_v.device).unsqueeze(0) # (1, num_knots_v, embd_dim)
            knots_pe_u[i, :num_knots_u[i], :] = _knots_pe_u_i
            knots_pe_v[i, :num_knots_v[i], :] = _knots_pe_v_i

        knots_tokens_u = knots_tokens_u + knots_pe_u
        knots_tokens_v = knots_tokens_v + knots_pe_v


        knots_u = self.knots_head_u(knots_tokens_u, padding_mask = knots_padding_mask_u).squeeze(dim=-1) # (B, max_num_u_knots)
        knots_u = self.softplus(knots_u)
        knots_v = self.knots_head_v(knots_tokens_v, padding_mask = knots_padding_mask_v).squeeze(dim=-1) # (B, max_num_v_knots)
        knots_v = self.softplus(knots_v)



        mults_logits_u = self.mults_head_u(knots_tokens_u, padding_mask = knots_padding_mask_u) # (B, max_num_u_knots, max_degree + 1)
        mults_logits_v = self.mults_head_v(knots_tokens_v, padding_mask = knots_padding_mask_v) # (B, max_num_v_knots, max_degree + 1)

        return knots_u, knots_v, mults_logits_u, mults_logits_v


    def decode(self, z, num_knots_u, num_knots_v, num_poles_u, num_poles_v):
        
        # Input latent code and surface information, get the reconstructed parameters.
        # We assume to know the ground truth number of knots and poles.
        # Mainly for teacher forcing during training. 
        # z: (B, embd_dim)
        # u_degree, v_degree, num_u_knots, num_v_knots: (B, 1)
        B = z.shape[0]
        device = z.device
        deg_logits_u = self.deg_head_u(z) # (B, max_degree_u)
        deg_logits_v = self.deg_head_v(z) # (B, max_degree_v)
        peri_logits_u = self.peri_head_u(z) # (B, 1)
        peri_logits_v = self.peri_head_v(z) # (B, 1)
        knots_num_logits_u = self.knots_num_head_u(z) # (B, max_num_knots_u)
        knots_num_logits_v = self.knots_num_head_v(z) # (B, max_num_knots_v)


        knots_u, knots_v, mults_logits_u, mults_logits_v = self.decode_knots_mults(z, num_knots_u, num_knots_v)


        
        poles_pe_padded = torch.zeros(B, self.max_num_u_poles, self.max_num_v_poles, self.embd_dim, device=device)
        poles_mask = torch.zeros(B, self.max_num_u_poles, self.max_num_v_poles, device=device, dtype=torch.bool)
        for i in range(B):
            _poles_pe_i = self.poles_pe(num_poles_u[i], num_poles_v[i], device) # (h, w, embd_dim)
            poles_pe_padded[i, :num_poles_u[i], :num_poles_v[i], :] = _poles_pe_i
            poles_mask[i, :num_poles_u[i], :num_poles_v[i]] = 1
        poles_padding_mask = ~poles_mask
        poles_padding_mask = einops.rearrange(poles_padding_mask, 'b h w -> b (h w)')
        

        # poles_tokens = self.token_proj_poles(z).unsqueeze(1) # (B, 1, embd_dim)

        pole_queries, pole_mask = self.poles_query_generator(
            batch_size=B,
            target_h_list=num_poles_u,
            target_w_list=num_poles_v,
            max_h=self.max_num_u_poles,
            max_w=self.max_num_v_poles,
            device=device
        )

        # poles_tokens = poles_tokens.expand(-1, self.max_num_u_poles * self.max_num_v_poles, -1)
        # poles_tokens = einops.rearrange(poles_tokens, 'b (n m) d -> b n m d', n=self.max_num_u_poles, m=self.max_num_v_poles) # (B, max_num_u_poles, max_num_v_poles, embd_dim)
        
        poles_tokens = pole_queries + poles_pe_padded

        poles_tokens = poles_tokens.view(B, -1, self.embd_dim) # (B, max_num_u_poles * max_num_v_poles, embd_dim)

        poles = self.poles_head(poles_tokens, padding_mask=poles_padding_mask) # (B, max_num_u_poles, max_num_v_poles, 4)
        poles = einops.rearrange(poles, 'b (n m) d -> b n m d', n=self.max_num_u_poles, m=self.max_num_v_poles) # (B, num_poles_u, num_poles_v, 4)
        poles = poles * poles_mask.unsqueeze(-1)


        return deg_logits_u, deg_logits_v, peri_logits_u, peri_logits_v, knots_num_logits_u, knots_num_logits_v, knots_u, knots_v, mults_logits_u, mults_logits_v, poles


    def classify(self, z):

        B = z.shape[0]
        device = z.device
        deg_logits_u = self.deg_head_u(z) # (B, max_degree_u)
        deg_logits_v = self.deg_head_v(z) # (B, max_degree_v)
        peri_logits_u = self.peri_head_u(z) # (B, 1)
        peri_logits_v = self.peri_head_v(z) # (B, 1)
        knots_num_logits_u = self.knots_num_head_u(z) # (B, max_num_knots_u)
        knots_num_logits_v = self.knots_num_head_v(z) # (B, max_num_knots_v)
        

        pred_degree_u = torch.argmax(deg_logits_u, dim=-1) + 1 # (B, 1)
        pred_degree_v = torch.argmax(deg_logits_v, dim=-1) + 1 # (B, 1)
        pred_periodic_u = torch.sigmoid(peri_logits_u) > 0.5 # (B, 1)
        pred_periodic_v = torch.sigmoid(peri_logits_v) > 0.5 # (B, 1)
        pred_knots_num_u = torch.argmax(knots_num_logits_u, dim=-1)  # (B, 1)
        pred_knots_num_v = torch.argmax(knots_num_logits_v, dim=-1)  # (B, 1)

        knots_u, knots_v, mults_logits_u, mults_logits_v = self.decode_knots_mults(z, pred_knots_num_u, pred_knots_num_v)

        pred_mults_u = torch.argmax(mults_logits_u, dim=-1) + 1 # (B, max_num_u_knots)
        pred_mults_v = torch.argmax(mults_logits_v, dim=-1) + 1 # (B, max_num_v_knots)

        knots_mask_u = torch.arange(self.max_num_u_knots, device=device).unsqueeze(0) < pred_knots_num_u.unsqueeze(1) # (B, 1) 
        knots_mask_v = torch.arange(self.max_num_v_knots, device=device).unsqueeze(0) < pred_knots_num_v.unsqueeze(1) # (B, 1)


        pred_mults_u = pred_mults_u * knots_mask_u # (B, max_num_u_knots)
        pred_mults_v = pred_mults_v * knots_mask_v # (B, max_num_v_knots)
        non_peri_num_poles_u = pred_mults_u.sum(dim=-1) - pred_degree_u.squeeze(-1) - 1
        non_peri_num_poles_v = pred_mults_v.sum(dim=-1) - pred_degree_v.squeeze(-1) - 1
        peri_num_poles_u = pred_mults_u[..., 1:].sum(dim=-1)
        peri_num_poles_v = pred_mults_v[..., 1:].sum(dim=-1)
        num_poles_u = []
        num_poles_v = []
        for i in range(B):
            if pred_periodic_u[i]:
                num_poles_u.append(peri_num_poles_u[i])
            else:
                num_poles_u.append(non_peri_num_poles_u[i])
            if pred_periodic_v[i]:
                num_poles_v.append(peri_num_poles_v[i])
            else:
                num_poles_v.append(non_peri_num_poles_v[i])
        num_poles_u = torch.tensor(num_poles_u, device=device).clamp(min=1, max=self.max_num_u_poles)
        num_poles_v = torch.tensor(num_poles_v, device=device).clamp(min=1, max=self.max_num_v_poles)


        return pred_degree_u, pred_degree_v, pred_periodic_u, pred_periodic_v, pred_knots_num_u, pred_knots_num_v, pred_mults_u, pred_mults_v, num_poles_u, num_poles_v


    def inference(self, z):
        # Here we don't know all the scalar information, need to infer from the latent code.
        pred_degree_u, pred_degree_v, pred_periodic_u, pred_periodic_v, pred_knots_num_u, pred_knots_num_v, pred_mults_u, pred_mults_v, pred_num_poles_u, pred_num_poles_v = self.classify(z)
        deg_logits_u, deg_logits_v, peri_logits_u, peri_logits_v, knots_num_logits_u, knots_num_logits_v, knots_u, knots_v, mults_logits_u, mults_logits_v, poles = self.decode(z, pred_knots_num_u, pred_knots_num_v, pred_num_poles_u, pred_num_poles_v)
        # surface_type = torch.argmax(surface_logits, dim=-1)
        return pred_degree_u, pred_degree_v, pred_periodic_u, pred_periodic_v, pred_knots_num_u, pred_knots_num_v, pred_mults_u, pred_mults_v, pred_num_poles_u, pred_num_poles_v, knots_u, knots_v, poles


if __name__ == "__main__":

    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.dataset.dataset_bspline import dataset_bspline
    from torch.utils.data import DataLoader
    import torch

    import numpy as np
    import einops
    from tqdm import tqdm

    mse_loss = torch.nn.MSELoss(reduction='none')
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
    bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

    model = BSplineVAE()
    dataset = dataset_bspline(data_path='/home/qindafei/CAD/data/logan_bspline/0/0000')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)


    for data in tqdm(dataloader):
        forward_args = data
        valid = forward_args[-1].bool()
        forward_args = [_[valid] for _ in forward_args[:-1]]
        forward_args = [_.unsqueeze(-1) if len(_.shape) == 1 else _ for _ in forward_args]
        forward_args = [_.float() if _.dtype == torch.float64 else _ for _ in forward_args]
        u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v, is_u_periodic, is_v_periodic, u_knots_list, v_knots_list, u_mults_list, v_mults_list, poles = forward_args
        B = u_degree.shape[0]
        if B == 0:
            # No valid data in this batch
            continue
        u_mults_list = u_mults_list.long()
        v_mults_list = v_mults_list.long()
        u_degree -= 1 # Start from 0
        v_degree -= 1 # Start from 0
        u_mults_list[u_mults_list > 0] -= 1 # Start from 0
        v_mults_list[v_mults_list > 0] -= 1 # Start form 0
        num_knots_u = num_knots_u.long()
        num_knots_v = num_knots_v.long()
        num_poles_u = num_poles_u.long()
        num_poles_v = num_poles_v.long()
        mu, logvar = model.encode(u_knots_list, u_mults_list, v_knots_list, v_mults_list, poles, u_degree, v_degree, is_u_periodic, is_v_periodic, num_knots_u, num_knots_v, num_poles_u, num_poles_v)
        z = model.reparameterize(mu, logvar)
        deg_logits_u, deg_logits_v, peri_logits_u, peri_logits_v, knots_num_logits_u, knots_num_logits_v, pred_knots_u, pred_knots_v, mults_logits_u, mults_logits_v, pred_poles = model.decode(z, num_knots_u, num_knots_v, num_poles_u, num_poles_v)
        
        # No need for mask
        loss_deg_u = ce_loss(deg_logits_u, u_degree.squeeze(-1)).mean() # Cross Entropy Loss for degree, max_degree - 1
        loss_deg_v = ce_loss(deg_logits_v, v_degree.squeeze(-1)).mean() # Cross Entropy Loss for degree, max_degree - 1
        loss_peri_u = bce_logits_loss(peri_logits_u.squeeze(-1), is_u_periodic.squeeze(-1).float()).mean() # Binary Cross Entropy Loss for periodic
        loss_peri_v = bce_logits_loss(peri_logits_v.squeeze(-1), is_v_periodic.squeeze(-1).float()).mean() # Binary Cross Entropy Loss for periodic
        loss_knots_num_u = ce_loss(knots_num_logits_u, num_knots_u.squeeze(-1)).mean() # Mean Squared Error Loss for number of knots
        loss_knots_num_v = ce_loss(knots_num_logits_v, num_knots_v.squeeze(-1)).mean() # Mean Squared Error Loss for number of knots

        # Need knots mask
        mask_u_knots = torch.arange(model.max_num_u_knots, device=num_knots_u.device).unsqueeze(0).repeat(num_knots_u.shape[0], 1) < num_knots_u # 1 for valid pos, 0 for invalid
        mask_v_knots = torch.arange(model.max_num_v_knots, device=num_knots_v.device).unsqueeze(0).repeat(num_knots_v.shape[0], 1) < num_knots_v # 1 for valid pos, 0 for invalid
        
        # Knots loss
        loss_knots_u = mse_loss(pred_knots_u, u_knots_list) # Mean Squared Error Loss for knots
        loss_knots_v = mse_loss(pred_knots_v, v_knots_list) # Mean Squared Error Loss for knots
        
        loss_knots_u = (loss_knots_u * mask_u_knots / num_knots_u).sum(dim=-1).mean() # Average over the valid positions
        loss_knots_v = (loss_knots_v * mask_v_knots / num_knots_v).sum(dim=-1).mean()

        # Mults loss
        mults_logits_u = mults_logits_u.view(-1, model.max_degree + 1)
        mults_logits_v = mults_logits_v.view(-1, model.max_degree + 1)
        u_mults_list = u_mults_list.view(-1)
        v_mults_list = v_mults_list.view(-1)
        loss_mults_u = ce_loss(mults_logits_u, u_mults_list) # Mean Squared Error Loss for mults
        loss_mults_v = ce_loss(mults_logits_v, v_mults_list) # Mean Squared Error Loss for mults

        loss_mults_u = loss_mults_u * mask_u_knots.view(-1)
        loss_mults_v = loss_mults_v * mask_v_knots.view(-1)
        loss_mults_u = loss_mults_u.sum() / mask_u_knots.sum()
        loss_mults_v = loss_mults_v.sum() / mask_v_knots.sum()



        # Need poles mask
        loss_poles = mse_loss(pred_poles, poles) # Mean Squared Error Loss for poles
        mask_poles_u = torch.arange(model.max_num_u_poles, device=num_poles_u.device).unsqueeze(0).repeat(num_poles_u.shape[0], 1) < num_poles_u # 1 for valid pos, 0 for invalid
        mask_poles_v = torch.arange(model.max_num_v_poles, device=num_poles_v.device).unsqueeze(0).repeat(num_poles_v.shape[0], 1) < num_poles_v # 1 for valid pos, 0 for invalid
        # loss_poles = (loss_poles * mask_poles_u.unsqueeze(2).unsqueeze(-1) / num_poles_u.unsqueeze(-1).unsqueeze(-1))
        # loss_poles = (loss_poles * mask_poles_v.unsqueeze(1).unsqueeze(-1) / num_poles_v.unsqueeze(-1).unsqueeze(-1)).sum(dim=(1, 2, 3)).mean()

        mask_poles_2d = mask_poles_u.unsqueeze(-1) & mask_poles_v.unsqueeze(-2)  # (B, H, W)
        mask_poles_4d = mask_poles_2d.unsqueeze(-1)  # (B, H, W, 1) → broadcast to 4 coords

        loss_poles = (loss_poles * mask_poles_4d).sum() / (mask_poles_4d.sum().clamp(min=1))

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        loss_total = loss_deg_u + loss_deg_v + loss_peri_u + loss_peri_v + loss_knots_num_u + loss_knots_num_v + loss_knots_u + loss_knots_v + loss_mults_u + loss_mults_v + loss_poles + kl_loss
        print(f"Loss: {loss_total.item()}, Loss Deg U: {loss_deg_u.item()}, Loss Deg V: {loss_deg_v.item()}, Loss Peri U: {loss_peri_u.item()}, Loss Peri V: {loss_peri_v.item()}, Loss Knots Num U: {loss_knots_num_u.item()}, Loss Knots Num V: {loss_knots_num_v.item()}, Loss Knots U: {loss_knots_u.item()}, Loss Knots V: {loss_knots_v.item()}, Loss Mults U: {loss_mults_u.item()}, Loss Mults V: {loss_mults_v.item()}, Loss Poles: {loss_poles.item()}, KL Loss: {kl_loss.item()}")



        pred_degree_u, pred_degree_v, pred_periodic_u, pred_periodic_v, pred_knots_num_u, pred_knots_num_v, pred_mults_u, pred_mults_v, pred_num_poles_u, pred_num_poles_v, knots_u, knots_v, poles = model.inference(z)

        print()
   