# src/air_quality_prediction/models/tab_transformer.py
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TabTransformer(nn.Module):
    """
    TabTransformer for regression (numerical features only).
    Based on: https://arxiv.org/abs/2012.06678

    Each numerical feature is projected to d_token-dimensional embedding.
    Tokens are processed by Transformer encoder.
    Final representation = mean of token embeddings.

    Args:
        n_num_features (int): number of numerical features
        d_token (int): embedding dim for each feature (d_model)
        n_layers (int): number of transformer layers
        n_heads (int): number of attention heads (must divide d_token)
        dropout (float): dropout rate
    """

    def __init__(
        self,
        n_num_features: int,
        d_token: int = 32,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if d_token % n_heads != 0:
            raise ValueError(f"d_token ({d_token}) must be divisible by n_heads ({n_heads})")

        self.n_num_features = n_num_features
        self.d_token = d_token

        self.feature_embedding = nn.Linear(1, d_token)
        self.embed_proj = nn.Linear(n_num_features, n_num_features * d_token)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.regressor = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """
        x_num: (batch_size, n_num_features)
        → (batch_size,)
        """
        B, L = x_num.shape

        x = self.embed_proj(x_num).view(B, L, self.d_token)
        x = x.permute(1, 0, 2)

        encoded = self.transformer(x)
        features = encoded.mean(dim=0)

        out = self.regressor(features).squeeze(-1)
        return out
