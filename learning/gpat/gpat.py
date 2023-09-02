
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from learning.pointnet import PointNet
from learning.modules import *

class SegmentationHead(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(SegmentationHead, self).__init__()
        self.Wq = FeedForward(hidden_dim=hidden_dim, dropout=dropout)
        self.Wk = FeedForward(hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, point_tokens, part_tokens, mask):
        query = self.Wq(point_tokens)
        key = self.Wk(part_tokens)
        d_k = query.size(-1)
        scores = torch.matmul(key, query.transpose(-2, -1)) / np.sqrt(d_k) # B, K, Q
        scores = scores.masked_fill(mask.unsqueeze(-1) == 1, -1e9).transpose(-2, -1) # B, Q, K
        return F.softmax(scores, dim = -1)

class TargetEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(TargetEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.pc_encoder = PointNet(out_dim=hidden_dim)
        self.downsample = Downsample(stride=10)
    
    def forward(self, pc):
        pc_encode = self.pc_encoder(pc) # B, N_tar, H
        return self.downsample(pc, pc_encode) # B, N_tar / downsample_rate, H

class PartEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(PartEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.pc_encoder = PointNet(out_dim=hidden_dim-1)
    
    def forward(self, pc):
        B, K, N, _ = pc.shape
        pc_enc = self.pc_encoder(pc.reshape(B*K, N, 3)).max(dim=1)[0].reshape(B, K, -1)
        instance_embed = torch.empty(B, K, 1).to(pc.device) # B, K, 1
        for b in range(B): instance_embed[b] = torch.randperm(K).unsqueeze(1).to(pc.device) + 1
        enc = torch.cat([pc_enc, instance_embed], dim=2) # B, K, H
        return enc

class GPAT(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0):
        super(GPAT, self).__init__()
        self.target_enc = TargetEncoder(hidden_dim=hidden_dim)
        self.part_enc = PartEncoder(hidden_dim=hidden_dim)
        n_neighbors = [16, 16, 32, 32, 64, 64]
        layers = [AttentionLayer(hidden_dim=hidden_dim, num_neighbors=num_neighbors, dropout=dropout) for num_neighbors in n_neighbors]
        layers.append(AttentionLayer(hidden_dim=hidden_dim, use_tf=True, dropout=dropout))
        layers.append(AttentionLayer(hidden_dim=hidden_dim, use_tf=True, dropout=dropout))
        self.layers = nn.ModuleList(layers)
        self.classifier = SegmentationHead(hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, part_pcs, target_pc, mask):
        point_tokens, points, point_idx = self.target_enc(target_pc)
        part_tokens = self.part_enc(part_pcs)
        for layer in self.layers:
            point_tokens, part_tokens = layer(points, point_tokens, part_tokens, mask)
        prob = self.classifier(point_tokens, part_tokens, mask)
        return prob, point_idx