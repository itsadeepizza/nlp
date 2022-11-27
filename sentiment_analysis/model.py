import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module): # TODO: multiple heads
  def __init__(self, Q_dim, K_dim, V_dim):
    super().__init__()

    self.Q_dim = Q_dim
    self.K_dim = K_dim
    self.V_dim = V_dim

    self.linear_Q = nn.Linear(self.Q_dim, self.Q_dim)
    self.linear_K = nn.Linear(self.K_dim, self.K_dim)
    self.linear_V = nn.Linear(self.V_dim, self.V_dim)

  def forward(self, Q, K, V):
    Q = self.linear_Q(Q)
    K = self.linear_K(K)
    V = self.linear_V(V)

    x = torch.matmul(Q, torch.transpose(K, -1, -2))
    x = x / torch.sqrt(torch.tensor([K.shape[-1]]))
    x = F.softmax(x, dim=-1)

    x = torch.matmul(x, V)

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

    self.encoder = nn.Sequential(
                                  PointwiseLayer(embedding_dim, 32),
                                  PointwiseLayer(32, 32),
                                  PointwiseLayer(32, 16)
                                )

    self.decoder = nn.Sequential(
                                  PointwiseLayer(16, 32),
                                  PointwiseLayer(32, 32),
                                  PointwiseLayer(32, embedding_dim)
                                 )

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x

class EncoderBlock(nn.Module):
  def __init__(self, embedding_dim):
    super().__init__()

    self.attention_layer = AttentionLayer(
                                            Q_dim=embedding_dim,
                                            K_dim=embedding_dim,
                                            V_dim=embedding_dim
                                         )

    self.layer_norm_0 = nn.LayerNorm(embedding_dim)
    self.pointwise_block_0 = PointwiseBlock(embedding_dim)
    self.layer_norm_1 = nn.LayerNorm(embedding_dim)


  def forward(self, x):
    x = self.layer_norm_0(self.attention_layer(x, x, x) + x)
    x = self.layer_norm_1(self.pointwise_block_0(x) + x)
    return x


if __name__ == "__main__":
    embedding_dim = 96
    encoder_block = EncoderBlock(embedding_dim)