import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_out = idx

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx_out


class SelfAttentionLayer(nn.Module):

    def __init__(self, channels):
        super(SelfAttentionLayer, self).__init__()

        self.bn_q = nn.BatchNorm1d(channels)
        self.bn_k = nn.BatchNorm1d(channels)
        self.bn_v = nn.BatchNorm1d(channels)
        self.conv_q = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                            self.bn_q,
                            nn.LeakyReLU(negative_slope=0.2))
        self.conv_k = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                            self.bn_k,
                            nn.LeakyReLU(negative_slope=0.2))
        self.conv_v = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                            self.bn_v,
                            nn.LeakyReLU(negative_slope=0.2))

        self.softmax = nn.Softmax(1)

        self.bn = nn.BatchNorm1d(channels)
        self.conv = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                                  self.bn,
                                  nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x_q = self.conv_q(x).permute(0, 2, 1)   # b, n, c
        x_k = self.conv_k(x)                    # b, c, n
        x_v = self.conv_v(x)

        attention = torch.bmm(x_q, x_k) / math.sqrt(x.size(1))  # b, n, n
        attention = self.softmax(attention)

        x_r = torch.bmm(x_v, attention)  # b, c, n

        x = x + self.conv(x_r)

        return x


class UprightNet(nn.Module):
    def __init__(self):
        super(UprightNet, self).__init__()

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False),
                            self.bn1,
                            nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                            self.bn2,
                            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                            self.bn3,
                            nn.LeakyReLU(negative_slope=0.2))

        self.sa1 = SelfAttentionLayer(128)
        self.sa2 = SelfAttentionLayer(128)
        self.sa3 = SelfAttentionLayer(128)
        self.sa4 = SelfAttentionLayer(128)

        self.bn4 = nn.BatchNorm1d(1024)
        self.conv4 = nn.Sequential(nn.Conv1d(128*4, 1024, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn5 = nn.BatchNorm1d(512)
        self.conv5 = nn.Sequential(nn.Conv1d(128+512+1024+1024, 512, kernel_size=1, bias=False),
                                  self.bn5,
                                  nn.LeakyReLU(negative_slope=0.2))
        self.bn6 = nn.BatchNorm1d(64)
        self.conv6 = nn.Sequential(nn.Conv1d(512, 64, kernel_size=1, bias=False),
                                  self.bn6,
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Conv1d(64, 1, kernel_size=1, bias=False)
        
        self.sm_fn = nn.Sigmoid()

    def forward(self, points):
        batch_size = points.size()[0]
        num_points = points.size()[2]

        x, _ = get_graph_feature(points, k=20)
        x = self.conv1(x)
        x = x.max(dim=-1, keepdim=False)[0]

        x, _ = get_graph_feature(x, k=20)
        x = self.conv2(x)
        x = x.max(dim=-1, keepdim=False)[0]

        x, _ = get_graph_feature(x, k=20)
        x = self.conv3(x)
        x_a = x.max(dim=-1, keepdim=False)[0]

        x1 = self.sa1(x_a)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        x_b = torch.cat((x1, x2, x3, x4), dim=1)

        x_c = self.conv4(x_b)
        global_feat = F.adaptive_max_pool1d(x_c, 1).view(batch_size, -1)

        x_global = global_feat.view(batch_size, -1, 1).repeat(1, 1, num_points)
        x = torch.cat((x_a, x_b, x_c, x_global), dim=1)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.sm_fn(x)

        return x
