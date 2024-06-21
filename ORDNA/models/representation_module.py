import torch
import einops
from torch import nn

import ORDNA.utils.constants as constants


class SelfAttentionRepresentationModule(nn.Module):
    def __init__(self, token_emb_dim: int, seq_len: int, repr_dim: int) -> None:
        super().__init__()

        self.token_emb_layer = nn.Embedding(num_embeddings=constants.NUM_TOKENS, embedding_dim=token_emb_dim)
        self.conv_1 = nn.Conv1d(in_channels=token_emb_dim, out_channels=2*token_emb_dim, kernel_size=5, padding=2)
        self.conv_2 = nn.Conv1d(in_channels=2*token_emb_dim, out_channels=4*token_emb_dim, 
                                kernel_size=5, dilation=5, padding=10)
        
        self.seq_mlp = nn.Sequential(nn.Linear(in_features=8*token_emb_dim*seq_len, out_features=2048),
                                     nn.ReLU(),
                                     nn.Linear(in_features=2048, out_features=repr_dim))

        self.sample_emb_layer = nn.Embedding(num_embeddings=1, embedding_dim=repr_dim)
        self.attention_1 = nn.MultiheadAttention(embed_dim=repr_dim, num_heads=4, batch_first=True)
        self.attention_2 = nn.MultiheadAttention(embed_dim=repr_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(normalized_shape=repr_dim)

    def forward(self, x: torch.Tensor):
        # B = Batch Size, N = #sequences, L = sequence length, T = token_emb_dim
        nucleotide_emb = self.token_emb_layer(x) # B x N x 2 x L x T
        B, N, _, _, _ = nucleotide_emb.size()
        forward, reverse = einops.rearrange(nucleotide_emb, 'B N dir L T -> dir (B N) T L', dir=2) # Both: B*N x T x L

        forward_emb = self.conv_1(forward) # B*N x 2*T x L
        reverse_emb = self.conv_1(reverse) # B*N x 2*T x L

        forward_emb = self.conv_2(forward_emb) # B*N x 4*T x L
        reverse_emb = self.conv_2(reverse_emb) # B*N x 4*T x L

        seq_emb = einops.rearrange([forward_emb, reverse_emb], 'dir BN T L -> BN (dir T L)', dir=2)
        seq_emb = self.seq_mlp(seq_emb)
        seq_emb = einops.rearrange(seq_emb, '(B N) out_dim -> B N out_dim', B=B, N=N)

        zero_tensor = torch.zeros(B, device=x.device, dtype=torch.int)
        sample_repr = self.sample_emb_layer(zero_tensor).unsqueeze(1) # B x 1 x repr_dim
        emb = torch.cat(tensors=(sample_repr, seq_emb), dim=1) # B x sample_subset_size+1 x repr_dim

        attn_output_1, _ = self.attention_1(emb, emb, emb, need_weights=False) # B x sample_subset_size+1 x repr_dim
        attn_output_1 = self.layer_norm(attn_output_1 + emb)

        attn_output_2, _ = self.attention_2(attn_output_1, attn_output_1, attn_output_1, need_weights=False)
        attn_output_2 = self.layer_norm(attn_output_2 + attn_output_1)

        sample_repr = attn_output_2[:, 0] # B x repr_dim

        return sample_repr
