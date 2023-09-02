
import torch
from torch import nn
import copy
from utils.pointops.functions import pointops

class Downsample(nn.Module):
    
    def __init__(self,
                 stride=4,
                 num_neighbors=16):
        assert stride > 1
        super(Downsample, self).__init__()        
        self.stride = stride
        self.grouper = pointops.QueryAndGroup(nsample=num_neighbors, use_xyz=False)
        
    def forward(self, p1, x):
        N = x.shape[1]
        M = N // self.stride
        p1_trans = p1.transpose(1, 2).contiguous()
        x_trans = x.transpose(1, 2).contiguous()
        idx = pointops.furthestsampling(p1, M)
        p2 = pointops.gathering(p1_trans, idx).transpose(1, 2).contiguous()
        n_x, _ = self.grouper(xyz=p1, new_xyz=p2, features=x_trans)
        y = n_x.mean(dim=3).transpose(1, 2).contiguous()
        return y, p2, idx

class NNAttention(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 num_neighbors=16):
        super(NNAttention, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        self.to_query = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
        self.to_key = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
        self.to_value = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
        self.to_pos_enc = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, self.out_channels, kernel_size=1)
        )
        self.to_attn = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        )
        self.key_grouper = pointops.QueryAndGroup(nsample=num_neighbors, return_idx=True)
        self.value_grouper = pointops.QueryAndGroup(nsample=num_neighbors, use_xyz=False)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, p, x):
        # p: (B, N, 3), x: (B, C_in, N)
        q = self.to_query(x) # (B, C_out, N)
        k = self.to_key(x) # (B, C_out, N)
        v = self.to_value(x) # (B, C_out, N)
        n_k, _, n_idx = self.key_grouper(xyz=p, features=k) # (B, 3+C_out, N, K)
        n_v, _ = self.value_grouper(xyz=p, features=v, idx=n_idx.int()) # (B, C_out, N, K)
        n_r = self.to_pos_enc(n_k[:, 0:3, :, :]) # (B, C_out, N, K)
        n_v = n_v + n_r
        a = self.to_attn(q.unsqueeze(-1) - n_k[:, 3:, :, :] + n_r) # (B, C_out, N, K)
        a = self.softmax(a)
        y = torch.sum(n_v * a, dim=-1, keepdim=False)
        return y
    
class NNAttentionBlock(nn.Module):
    
    def __init__(self,
                 in_channels,
                 dropout,
                 out_channels=None,
                 num_neighbors=16):
        super(NNAttentionBlock, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        
        self.linear1 = nn.Conv1d(in_channels, self.out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.transformer = NNAttention(self.out_channels, num_neighbors=num_neighbors)
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.linear2 = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, p, x):
        x = x.transpose(-2, -1).contiguous()
        y = self.relu(self.bn1(self.dropout(self.linear1(x))))
        y = self.relu(self.bn(self.dropout(self.transformer(p, y))))
        y = self.bn2(self.dropout(self.linear2(y)))
        y += x
        y = self.relu(y).transpose(-2, -1).contiguous()
        return y

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PointAttentionTF(nn.Module):
    def __init__(self, hidden_dim, dropout, nhead=4):
        super(PointAttentionTF, self).__init__()
        self.model = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
    
    def forward(self, points, point_features):
        return self.model(point_features)

class PointAttention(nn.Module):
    def __init__(self, hidden_dim, num_neighbors, dropout):
        super(PointAttention, self).__init__()
        self.model = NNAttentionBlock(in_channels=hidden_dim, dropout=dropout, num_neighbors=num_neighbors)
    
    def forward(self, points, point_features):
        return self.model(points, point_features)

class PartAttention(nn.Module):
    def __init__(self, hidden_dim, dropout, nhead=4):
        super(PartAttention, self).__init__()
        self.model = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
    
    def forward(self, part_features, mask):
        return self.model(part_features, src_key_padding_mask=mask)

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(FeedForward, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(hidden_dim*4, hidden_dim),
        )
    def forward(self, x):
        return self.model(x)

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, dropout, num_heads=4):
        super(CrossAttention, self).__init__()
        self.att = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(hidden_dim=hidden_dim, dropout=dropout)
        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_tokens, kv_tokens, mask=None):
        attn_output, attn_output_weights = self.att(query_tokens, kv_tokens, kv_tokens, mask)
        query_tokens = self.norm0(query_tokens + self.dropout(attn_output))
        query_tokens = self.norm1(query_tokens + self.dropout(self.ff(query_tokens)))
        return query_tokens

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, dropout, num_neighbors=None, num_heads=4, use_tf=False):
        super(AttentionLayer, self).__init__()
        if use_tf:
            self.point_attention = PointAttentionTF(hidden_dim, dropout=dropout)
        else:
            self.point_attention = PointAttention(hidden_dim, num_neighbors, dropout=dropout)
        self.part_attention = PartAttention(hidden_dim, dropout=dropout)
        self.point_to_part_attention = CrossAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.part_to_point_attention = CrossAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
    
    def forward(self, points, point_tokens, part_tokens, mask):
        point_tokens_tf = self.point_attention(points, point_tokens)
        part_tokens_tf = self.part_attention(part_tokens, mask)
        point_tokens_cross = self.point_to_part_attention(point_tokens_tf, part_tokens_tf, mask)
        part_tokens_cross = self.part_to_point_attention(part_tokens_tf, point_tokens_cross)
        return point_tokens_cross, part_tokens_cross