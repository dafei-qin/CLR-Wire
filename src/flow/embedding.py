import numpy as np
import torch
from torch import nn

# def centers(start: float, stop: float, num: int, device: torch.device | None = None) -> torch.Tensor:
def centers(start: float, stop: float, num: int, device=None) -> torch.Tensor:
    edges = torch.linspace(start, stop, num + 1, dtype=torch.float32, device=device)
    return (edges[:-1] + edges[1:]) / 2
    
class PointEmbed(nn.Module):
    # 3: xyz dim
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
        torch.cat([e, torch.zeros(self.embedding_dim // 6),
                torch.zeros(self.embedding_dim // 6)]),
        torch.cat([torch.zeros(self.embedding_dim // 6), e,
            torch.zeros(self.embedding_dim // 6)]),
        torch.cat([torch.zeros(self.embedding_dim // 6),
            torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer("basis", e) # 3 x 24

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        # (B, N, 3) --> (B, N, hidden_dim)
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        # output: B x N x dim
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed 


class PointEmbd2D(nn.Module):
    # 2: xyz dim
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 4 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 4)).float() * np.pi
        e = torch.stack([
        torch.cat([e, torch.zeros(self.embedding_dim // 4),]),
        torch.cat([torch.zeros(self.embedding_dim // 4), e,]),
        ])

        self.register_buffer("basis", e) # 2 x 24

        self.mlp = nn.Linear(self.embedding_dim + 2, dim)

    @staticmethod
    def embed(input, basis):
        # (B, N, 2) --> (B, N, hidden_dim)
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 2
        # output: B x N x dim
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed 

if __name__ == '__main__':
    pe = PointEmbd2D()
    data = torch.zeros((8,512, 2))
    embd = pe(data)
