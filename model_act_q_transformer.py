# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformer import Transformer, TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer


"""
model architecture: AQT (patch size: 7*7)
"""


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)



class AQT(nn.Module):
  def __init__(self, args, action_space, head_dim=32, num_encoder_layers=1, num_decoder_layers=1):
    super(AQT, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space
    hidden_dim = self.action_space * head_dim

    # feature extractor
    self.convs = nn.Sequential(
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1), nn.ReLU()
        )
    
    # transformer encoder
    encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                            dropout=0.1, activation="relu", normalize_before=False)
    self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
    self.encoder_output_size = 7 * 7 * hidden_dim

    # transformer decoder
    decoder_layer = TransformerDecoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                            dropout=0.1, activation="relu", normalize_before=False)
    decoder_norm = nn.LayerNorm(hidden_dim)
    self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)  
    self.decoder_output_size = hidden_dim      
    
    # query branch
    self.act_list = torch.zeros(self.action_space, self.action_space, device=args.device)
    for i in range(self.action_space):
        self.act_list[i][i] = 1.0
    self.action_encoder = nn.Linear(self.action_space, hidden_dim)

    # positional encodings
    self.row_embed = nn.Parameter(torch.rand(7, hidden_dim // 2))
    self.col_embed = nn.Parameter(torch.rand(7, hidden_dim // 2))
    
    # value branch
    self.fc_h_v = NoisyLinear(self.encoder_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.decoder_output_size, args.hidden_size, std_init=args.noisy_std)

    # advantage branch
    self.fc_h_a = NoisyLinear(self.decoder_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)

    self._reset_parameters()


  def _reset_parameters(self):
      for p in self.parameters():
          if p.dim() > 1:
              nn.init.xavier_uniform_(p)

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()
  
  def a_reset_noise(self):
    for name, module in self.named_children():
      if ('fc' in name) and ('_a' in name):
        module.reset_noise()


  def forward(self, x, log=False):
    x = self.convs(x)
    
    # transformer encoder
    bs, c, h, w = x.shape
    pos = torch.cat([
        self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1),
        self.row_embed[:h].unsqueeze(1).repeat(1, w, 1),
    ], dim=-1).flatten(0, 1).unsqueeze(1)
    pos = pos.expand(pos.shape[0], bs, pos.shape[2])
    src = x.flatten(2).permute(2, 0, 1)
    memory = self.transformer_encoder(pos + 0.1 * src)

    # value head
    v = self.fc_z_v(F.relu(self.fc_h_v(memory.permute(1, 2, 0).reshape(-1, self.encoder_output_size))))

    # action queries
    query_embed = []
    for action in self.act_list:
        action_query = self.action_encoder(action).unsqueeze(0)
        query_embed.append(action_query)
    query_embed = torch.cat(query_embed, dim=0)

    # transformer decoder
    tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
    hs = self.transformer_decoder(tgt, memory)[0] # 6,batch,192
    hs = hs.permute(1,0,2) # batch,6,192

    # advantage head
    adv = self.fc_z_a(F.relu(self.fc_h_a(hs))) # batch,6,51

    # Combine streams
    v, adv = v.view(-1, 1, self.atoms), adv.view(-1, self.action_space, self.atoms) # batch,1,51 / batch,6,51
    q = v + adv - adv.mean(1, keepdim=True)
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q, v, adv
