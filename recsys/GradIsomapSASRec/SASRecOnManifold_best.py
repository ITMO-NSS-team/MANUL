"""
SASRec with item representations from IsomapNN manifold coordinates Z.

Mirrors NeuMFOnManifold's design exactly:
  - item_projection: nn.Linear(latent_dim, hidden_units) replaces item_emb
  - freeze_item_projection: fixed PCA projection (data-informed, not random),
    only positional embedding + transformer weights remain trainable
  - item_projection_init_data: current item_Z for PCA initialization
  - _init_weights: same guard logic as NeuMFOnManifold

Key SASRec-specific note on freezing:
  NeuMFOnManifold has two branches (GMF=PCA, MLP=identity).
  SASRec has one item_projection: when frozen, it uses PCA to hidden_units.
  The identity trick (latent_dim == mlp_user_dim) doesn't apply here —
  SASRec's hidden_units is independent of latent_dim by design.
"""
import numpy as np
import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    """Identical to source.py PointWiseFeedForward."""
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASRecOnManifold(nn.Module):
    """
    SASRec where item_emb (nn.Embedding) is replaced by item_projection
    (nn.Linear(latent_dim, hidden_units)), sourced from IsomapNN Z.

    Parameters
    ----------
    config : dict
        hidden_units, maxlen, num_blocks, num_heads, dropout_rate, l2_emb
    item_num : int
        Number of real items. pad_token = item_num (same as source.py).
    latent_dim : int
        Dimension of Z from IsomapNN.
    freeze_item_projection : bool
        If True, item_projection is fixed (PCA of item_Z). Only positional
        embedding and transformer blocks remain trainable.
        Mirrors NeuMFOnManifold.freeze_item_projection.
    item_projection_init_data : torch.Tensor | None
        item_Z matrix [num_items, latent_dim] used to fit the fixed PCA
        projection. Required when freeze_item_projection=True.
    """

    def __init__(
        self,
        config: dict,
        item_num: int,
        latent_dim: int,
        freeze_item_projection: bool = False,
        item_projection_init_data: "torch.Tensor | None" = None,
    ):
        super().__init__()
        self.item_num = item_num
        self.pad_token = item_num          # same convention as source.py
        self.latent_dim = latent_dim
        self.hidden_units = config['hidden_units']
        self.freeze_item_projection = freeze_item_projection
        self.config = config

        # ── Item projection: Z → hidden_units (replaces nn.Embedding) ──
        self.item_projection = nn.Linear(latent_dim, self.hidden_units)

        # ── Positional embedding (identical to source.py) ──
        self.pos_emb = nn.Embedding(config['maxlen'], self.hidden_units)
        self.emb_dropout = nn.Dropout(p=config['dropout_rate'])

        # ── Transformer blocks (identical to source.py) ──
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)

        for _ in range(config['num_blocks']):
            self.attention_layernorms.append(
                nn.LayerNorm(self.hidden_units, eps=1e-8)
            )
            self.attention_layers.append(
                nn.MultiheadAttention(
                    self.hidden_units,
                    config['num_heads'],
                    config['dropout_rate'],
                )
            )
            self.forward_layernorms.append(
                nn.LayerNorm(self.hidden_units, eps=1e-8)
            )
            self.forward_layers.append(
                PointWiseFeedForward(self.hidden_units, config['dropout_rate'])
            )

        self._init_weights(latent_dim, item_projection_init_data)

    def _init_weights(self, latent_dim, item_projection_init_data):
        """
        Mirrors NeuMFOnManifold._init_weights() exactly.

        freeze_item_projection=True:
          - item_projection gets a fixed PCA projection (top-hidden_units
            principal directions of item_Z) — data-informed, not random.
          - All item_projection parameters are frozen (requires_grad=False).
          - pos_emb + transformer blocks remain trainable.

        freeze_item_projection=False:
          - item_projection: xavier_uniform (same as NeuMFOnManifold).
          - All parameters trainable.
        """
        if self.freeze_item_projection:
            if item_projection_init_data is None:
                raise ValueError(
                    "freeze_item_projection=True requires item_projection_init_data "
                    "(the item_Z matrix) to fit a fixed, well-conditioned PCA "
                    "projection — a random frozen projection can compress poorly."
                )

            with torch.no_grad():
                z = item_projection_init_data.detach().to(torch.float32)
                z_centered = z - z.mean(dim=0, keepdim=True)

                # Top-hidden_units principal directions via SVD.
                # Mirrors the GMF branch of NeuMFOnManifold exactly.
                _, _, Vt = torch.linalg.svd(z_centered, full_matrices=False)
                # Vt: [min(n,d), d] — rows are principal directions
                n_components = min(self.hidden_units, Vt.shape[0])
                projection = Vt[:n_components]         # [n_components, latent_dim]

                if n_components < self.hidden_units:
                    # Pad with zeros if fewer components than hidden_units
                    pad = torch.zeros(
                        self.hidden_units - n_components, latent_dim,
                        dtype=projection.dtype
                    )
                    projection = torch.cat([projection, pad], dim=0)

                self.item_projection.weight.copy_(projection)
                self.item_projection.bias.zero_()

            # Freeze item_projection — identical to NeuMFOnManifold pattern
            self.item_projection.weight.requires_grad_(False)
            self.item_projection.bias.requires_grad_(False)

        else:
            # Trainable projection — xavier_uniform, same as NeuMFOnManifold
            nn.init.xavier_uniform_(self.item_projection.weight)
            nn.init.zeros_(self.item_projection.bias)

        # Positional embedding and transformer: xavier_uniform (source.py style)
        for name, param in self.named_parameters():
            if 'item_projection' in name:
                continue   # already handled above
            try:
                nn.init.xavier_uniform_(param.data)
            except Exception:
                pass       # LayerNorm bias, 1-d params — skip silently

    def _embed_items(
        self,
        item_ids: torch.Tensor,    # any shape [...] of item indices
        item_Z: torch.Tensor,      # [num_items+1, latent_dim]
    ) -> torch.Tensor:
        """
        item_Z[pad_token] must be a zero row (added by pad_item_Z()).
        item_projection(zero_row) → zero vector, so padding is preserved.
        """
        return self.item_projection(item_Z[item_ids])  # [..., hidden_units]

    def log2feats(
        self,
        log_seqs: torch.Tensor,    # [B, L] — item indices
        item_Z: torch.Tensor,      # [num_items+1, latent_dim]
    ) -> torch.Tensor:
        """
        Mirrors SASRec.log2feats() exactly, sourcing item vectors from
        item_projection(item_Z[...]) instead of item_emb(log_seqs).
        """
        device = log_seqs.device
        B, L = log_seqs.shape

        # Item embeddings from manifold
        seqs = self._embed_items(log_seqs, item_Z)   # [B, L, H]
        seqs *= self.hidden_units ** 0.5              # scaling (source.py)

        # Positional embeddings
        positions = np.tile(np.arange(L), [B, 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        seqs = self.emb_dropout(seqs)

        # Padding mask: True where pad_token
        timeline_mask = (log_seqs == self.pad_token)  # [B, L]
        seqs *= ~timeline_mask.unsqueeze(-1)

        # Causal attention mask (True = blocked)
        attention_mask = ~torch.tril(
            torch.ones(L, L, dtype=torch.bool, device=device)
        )

        # Transformer blocks (identical to source.py)
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)        # [L, B, H]
            Q = self.attention_layernorms[i](seqs)
            mha_out, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )
            seqs = Q + mha_out
            seqs = torch.transpose(seqs, 0, 1)        # [B, L, H]

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        return self.last_layernorm(seqs)              # [B, L, H]

    def forward(
        self,
        log_seqs: torch.Tensor,    # [B, L]
        pos_seqs: torch.Tensor,    # [B, L]
        neg_seqs: torch.Tensor,    # [B, L]
        item_Z: torch.Tensor,      # [num_items+1, latent_dim]
    ):
        """
        Mirrors SASRec.forward() — returns (pos_logits, neg_logits), [B, L].
        Used in training loop for BCE loss on non-padding positions.
        """
        log_feats = self.log2feats(log_seqs, item_Z)   # [B, L, H]
        pos_embs = self._embed_items(pos_seqs, item_Z)  # [B, L, H]
        neg_embs = self._embed_items(neg_seqs, item_Z)  # [B, L, H]
        pos_logits = (log_feats * pos_embs).sum(dim=-1) # [B, L]
        neg_logits = (log_feats * neg_embs).sum(dim=-1) # [B, L]
        return pos_logits, neg_logits

    def predict_candidates(
        self,
        log_seqs: torch.Tensor,      # [B, maxlen]
        candidates: torch.Tensor,    # [B, C]
        item_Z: torch.Tensor,        # [num_items+1, latent_dim]
    ) -> torch.Tensor:
        """
        Score a fixed candidate set. Used in sampled-ranking evaluation.
        Returns [B, C] logits.
        """
        log_feats = self.log2feats(log_seqs, item_Z)   # [B, L, H]
        final_feat = log_feats[:, -1, :]               # [B, H]
        cand_embs = self._embed_items(candidates, item_Z)  # [B, C, H]
        return (final_feat.unsqueeze(1) * cand_embs).sum(dim=-1)  # [B, C]


def pad_item_Z(item_Z: torch.Tensor) -> torch.Tensor:
    """
    Append a zero row at index num_items (= pad_token).
    item_projection(zero_row) → zero vector → padding contributes nothing.
    Call once before passing item_Z to any SASRecOnManifold method.
    """
    pad_row = torch.zeros(1, item_Z.shape[1],
                          dtype=item_Z.dtype, device=item_Z.device)
    return torch.cat([item_Z, pad_row], dim=0)  # [num_items+1, latent_dim]
