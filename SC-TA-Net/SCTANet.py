import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import matplotlib

matplotlib.use('TkAgg')


def cwt(data, fs,k=4):
    """
    data: Tensor, shape [B, 1, T]
    return: List of 5 tensors, each [B, 1, k, 200]
    """
    wavename = "morl"
    totalscal = 51
    fc = 1
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)

    # 对应频率（长度为 50）
    frequencies = pywt.scale2frequency(wavename, scales) * fs  # shape: [50]

    # 频段划分
    band_ranges = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50),
    }

    output_list = {band: [] for band in band_ranges}

    B = data.shape[0]
    T = data.shape[-1]

    for i in range(B):
        sig = data[i, 0, :].cpu().numpy()
        cwtmatr, freqs = pywt.cwt(sig, scales, wavename, 1.0 / fs)  # shape: [50, T]
        cwtmatr = np.abs(cwtmatr)

        for band, (low, high) in band_ranges.items():
            idx = np.where((frequencies >= low) & (frequencies <= high))[0]
            band_tf = cwtmatr[idx, :]  # shape: [f', T]
            band_tf = torch.tensor(band_tf).unsqueeze(0).unsqueeze(0)  # [1, 1, f', T]
            band_tf = F.adaptive_avg_pool2d(band_tf, (10, 200))
            output_list[band].append(band_tf)

    # 拼接 batch
    for band in output_list:
        output_list[band] = torch.cat(output_list[band], dim=0).cuda()  # [B, 1, 10, 200]
        output_list[band].squeeze()

    return [
        output_list['delta'],
        output_list['theta'],
        output_list['alpha'],
        output_list['beta'],
        output_list['gamma']
    ]

class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_sizes=[16, 8, 4]):
        super(MultiScaleCNN, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2),
                nn.ReLU()
            ) for k in kernel_sizes
        ])
        self.output_proj = nn.Conv1d(len(kernel_sizes), 1, kernel_size=1)
        self.k1 = nn.Parameter(torch.tensor(1.0))
        self.k2 = nn.Parameter(torch.tensor(1.0))
        self.k3 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):  # x: [B, 1, 200]
        outs = [branch(x) for branch in self.branches]  # each: [B, 1, 200]
        outs[0] = outs[0]*self.k1
        outs[1] = outs[1]*self.k2
        outs[2] = outs[2]*self.k3
        out = torch.cat(outs, dim=1)  # [B, len(kernel_sizes), 200]
        out = self.output_proj(out)[:,:,0:200]  # [B, 1, 200]
        return out

class Encoder(nn.Module):
    def __init__(self, k):
        super(Encoder, self).__init__()
        self.conv_k = nn.Conv1d(in_channels=k, out_channels=1, kernel_size=1)
        self.conv_200 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):

        out = self.conv_k(x)  # [B, 1, 200]
        out = self.conv_200(out)  # [B, 1, 200]
        return out






class MCM(nn.Module):
    def __init__(self, total_channels, query_channel_index, d_model=200):
        super(MCM, self).__init__()
        self.query_channel_index = query_channel_index
        self.d_model = d_model
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.kv_weights = nn.Parameter(torch.randn(total_channels))
        mask = torch.ones(total_channels)
        mask[query_channel_index] = 0
        self.register_buffer("mask", mask)

    def forward(self, x):  # x: [B, C, 200]
        x= x.cuda()
        B, C, D = x.shape
        assert D == self.d_model, "Input last dim must match d_model"
        q = x[:, self.query_channel_index, :].unsqueeze(1).float()
        weights = F.softmax(self.kv_weights * self.mask, dim=0).cuda()
        kv = (weights.view(1, C, 1) * x).sum(dim=1, keepdim=True).float()
        out, _ = self.attn(q, kv, kv)  # [B, 1, 200]
        return out


class BAF(nn.Module):
    def __init__(self,k, num_bands=5, d_model=200):
        super(BAF, self).__init__()
        self.num_bands = num_bands
        self.k = k
        self.d_model = d_model

        self.band_selector = nn.Sequential(
            nn.Linear(num_bands * k * d_model, 512),
            nn.ReLU(),
            nn.Linear(512, num_bands),
            nn.Softmax(dim=-1)
        )

        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)

    def forward(self, bands):
        B = bands[0].shape[0]
        concat = torch.cat(bands, dim=1)
        selector_input = concat.reshape(B, -1)
        band_weights = self.band_selector(selector_input)
        selected_index = torch.argmax(band_weights, dim=-1)

        queries = []
        keys = []
        for b in range(B):
            idx = selected_index[b]
            q = bands[idx][b].unsqueeze(0)
            kv = [bands[i][b] for i in range(self.num_bands)]
            kv = torch.cat(kv, dim=0).unsqueeze(0)
            attn_output, _ = self.cross_attn(q, kv, kv)
            queries.append(attn_output)
        return torch.cat(queries, dim=0)


class FeatureFusion(nn.Module):
    def __init__(self, embed_dim1=200, embed_dim2=200, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim1 + embed_dim2
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(self.embed_dim, embed_dim1)
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=2)
        fused, attn_weights = self.attn(x, x, x)
        fused = self.proj(fused)
        return fused, attn_weights

class SCTANet(nn.Module):
    def __init__(self, num_class, k, query_channel_index, num_bands, num_channels):
        super(SCTANet, self).__init__()
        self.k = k
        self.query_channel_index = query_channel_index
        self.num_bands = num_bands
        self.cnn = MultiScaleCNN(in_channels=1, out_channels=1)
        self.band_fusion = BAF(num_bands=num_bands, k=k)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=200, nhead=4, batch_first=True),
            num_layers=2
        )

        self.Encoders = Encoder(10)
        self.channel_attention = MCM(total_channels=num_channels, query_channel_index=query_channel_index)
        self.fusion = FeatureFusion()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(200, num_class)
        )
        self.kk = nn.Parameter(torch.tensor(0.5))
        print('reading...')

        print('reading finish')

    def forward(self, x):
        x =x.unsqueeze(1)
        B,L,F = x.shape

        x_t = self.cnn(x)
        x_t = self.transformer(x_t.reshape(-1,self.k,200))

        x_f = cwt(x,200,k=self.k)
        xf = []
        for i in range(len(x_f)):
            xf.append(self.Encoders(x_f[i].reshape(-1,10,200)).reshape(-1,self.k,200))
        x_f_fused = self.band_fusion(xf)  # [B, k, 200]

        x_fused = self.fusion(x_f_fused, x_t).reshape(B,200)
        x_cls = self.classifier(x_fused)  # -> [B, num_class]

        return x_cls


