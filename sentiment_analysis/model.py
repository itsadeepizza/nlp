import torch
import torch.nn as nn
import torch.nn.functional as F
from config import selected_config as conf
import math


class AttentionLayer(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.linear_Q = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.linear_K = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.linear_V = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

    def forward(self, Q, K, V):
        Q = self.linear_Q(Q)
        K = self.linear_K(K)
        V = self.linear_V(V)

        x = torch.matmul(Q, torch.transpose(K, -1, -2))
        x = x / math.sqrt(self.embedding_dim)
        x = F.softmax(x, dim=-1)

        x = torch.matmul(x, V)

        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads):
        super().__init__()

        self.heads = nn.ModuleList([AttentionLayer(embedding_dim)] * n_heads)
        self.linear = nn.Linear(embedding_dim * n_heads, embedding_dim, bias=False)

    def forward(self, x):
        x = [head(x, x, x) for head in self.heads]
        x = torch.cat(x, dim=2)
        x = self.linear(x)

        return x


class PointwiseLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super().__init__()

        self.linear_0 = nn.Linear(input_dim, output_dim)
        self.dropout_0 = nn.Dropout(dropout_rate)
        self.layer_norm_0 = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = F.relu(self.linear_0(x))
        x = self.dropout_0(x)
        x = self.layer_norm_0(x)

        return x


class PointwiseBlock(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.encoder = nn.Sequential(PointwiseLayer(embedding_dim, 32), PointwiseLayer(32, 32), PointwiseLayer(32, 16))

        self.decoder = nn.Sequential(PointwiseLayer(16, 32), PointwiseLayer(32, 32), PointwiseLayer(32, embedding_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads):
        super().__init__()

        self.multi_head_attention_layer = MultiHeadAttentionLayer(embedding_dim, n_heads)

        self.layer_norm_0 = nn.LayerNorm(embedding_dim)
        self.pointwise_block_0 = PointwiseBlock(embedding_dim)

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.layer_norm_0(self.multi_head_attention_layer(x) + x)
        x = self.layer_norm_1(self.pointwise_block_0(x) + x)

        return x


class Embedding(nn.Module):

    def positional_encoding_token(self, position):
        dimension_index = torch.arange(self.embedding_dim)
        even_idx = torch.sin(position / (10000 ** (2 * (dimension_index) / self.embedding_dim)))
        odd_idx = torch.cos(position / (10000 ** (2 * (dimension_index) / self.embedding_dim)))
        odd_mask = (torch.ones(self.embedding_dim) - torch.pow(-1, dimension_index)) / 2
        even_mask = torch.ones(self.embedding_dim) - odd_mask
        pos_enc = even_idx * even_mask + odd_idx * odd_mask
        return pos_enc.to(conf.DEVICE)

    def positional_encoding_sentence(self, length_sentence):
        to_stack = [self.positional_encoding_token(idx_word) for idx_word in range(length_sentence)]
        return torch.stack(to_stack)

    def __init__(self, embedding_dim, max_id_token) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.embed_layer = nn.Embedding(max_id_token, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Embedding token
        x = self.embed_layer(x)
        # Positional Encoding
        positional_encoding = torch.stack([self.positional_encoding_sentence(x.shape[1])] * x.shape[0], axis=0)
        x = x + positional_encoding
        x = self.layer_norm(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()

        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.linear(x)

        return x


class Transformer(nn.Module):
    def __init__(self, embedding_dim=conf.EMBEDDING_DIM, n_heads=conf.N_HEADS, output_dim=conf.OUTPUT_DIM, max_id_token=conf.MAX_ID_TOKEN) -> None:
        super().__init__()

        self.embedding = Embedding(embedding_dim, max_id_token)
        self.encoder_stack = nn.Sequential(*[EncoderBlock(embedding_dim, n_heads)] * conf.N_ENCODER_BLOCK)  # conf.N_ENCODER_BLOCK

        self.classification_head = ClassificationHead(embedding_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder_stack(x)

        x = self.classification_head(x)  # NO SOFTMAX!!

        return x


if __name__ == "__main__":
    transformer = Transformer()
