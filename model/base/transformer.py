import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor
import os
import cv2

key_layer = 0

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, add_gaussian=False, add_bottle_layer=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.add_gaussian = add_gaussian
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn = None
        self.add_bottle_layer = add_bottle_layer
        if add_bottle_layer:
            self.bottle_layer = nn.Sequential(nn.Conv2d(h, 4*h, (3, 3), padding=(1, 1), bias=True),
                                            nn.ReLU(),
                                            nn.Conv2d(4*h, h, (3, 3), padding=(1, 1), bias=True),
                                            nn.ReLU())
        self.dropout = nn.Dropout(p=dropout)
        self.class_head = nn.Linear(self.h, 2)

    def forward(self, query, key, value, mask=None, nshot=1):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key = \
        #     [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #      for l, x in zip(self.linears, (query, key))]
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        # print(query.shape)
        # print(key.shape)
        query = rearrange(query, 'b p (head c) -> b head p c', head=self.h, c=self.d_k)
        key = rearrange(key, '(n b) p (head c) -> b head (n p) c', n=nshot, b=nbatches, head=self.h, c=self.d_k)
        value = value.repeat(self.h, 1, 1).transpose(0, 1).contiguous().unsqueeze(-1)
        # print(query.shape)
        # print(key.shape)
        # print(value.shape)

        # 2) Apply attention on all the projected vectors in batch.
        self.attn = attention(query, key, mask=mask, dropout=self.dropout, add_gaussian=self.add_gaussian, nshot=nshot)

        if self.add_bottle_layer:
            # print('bottle_layer')
            self.attn = self.bottle_layer(self.attn)

        x = torch.matmul(self.attn, value)

        # 3) "Concat" using a view and apply a final linear.
        return torch.mean(x, -3)
        # x = rearrange(x, 'b h p n -> b p (h n)')
        # x = self.class_head(x)
        # return x


class MultiHeadedAttentionMix(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, add_gaussian=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionMix, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.add_gaussian = add_gaussian
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.class_head = nn.Conv2d(self.h, 2, (1, 1), padding=(0, 0))

    def forward(self, query, key, value0, value1, mask=None, nshot=1):
        # print(query.shape)
        # print(key.shape)
        # print(value0.shape)
        # print(value1.shape)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key))]
        value0 = value0.repeat(self.h, 1, 1).transpose(0, 1).contiguous().unsqueeze(-1)
        value1 = value1.repeat(self.h, 1, 1).transpose(0, 1).contiguous().unsqueeze(-1)

        # 2) Apply attention on all the projected vectors in batch.
        self.attn = attention(query, key, mask=mask, dropout=self.dropout, add_gaussian=self.add_gaussian, nshot=nshot)
        x0 = torch.matmul(self.attn, value0)          # [heads hw 1]
        x1 = torch.matmul(self.attn.transpose(-1, -2), value1)      # [heads nhw 1]

        mask_similarity = x0@x1.transpose(-1, -2)        # [heads hw nhw]

        # 3) "Concat" using a view and apply a final linear.
        return torch.mean(mask_similarity, -3)
        # mask_similarity = self.class_head(mask_similarity)
        # return mask_similarity


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape)
        # print(self.pe.shape)
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def attention(query, key, mask=None, dropout=None, add_gaussian=False, nshot=1):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # global key_layer
    # key_layer += 1
    # if True:
    #     src = './data/mid_feature_map/dcama/img_similarity/layer' + str(key_layer) + '/'
    #     if not os.path.exists(src): os.mkdir(src)
    #     image_scores = torch.mean(scores, 1)
    #     for i in range(image_scores.shape[0]):
    #         img = image_scores[i]
    #         img_path = src + str(i) + '.jpg'
    #         save_feature_map(img, img_path)

    # print(scores.shape)
    if add_gaussian:
        scores = rearrange(scores, 'b head h (n w) -> (b n) head h w', n=nshot)
        scores = apply_gaussian_kernel(scores)
        scores = rearrange(scores, '(b n) head h w -> b head h (n w)', n=nshot)
    scores = scores/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn

def apply_gaussian_kernel(corr, sigma=5):
    b, head, H, W = corr.size()
    B = int(b * head)
    h = w = int(W**0.5)

    corr = rearrange(corr, 'b head hw (h1 w1) -> (b head) hw h1 w1', h1=h, w1=w)

    idx = corr.max(dim=1)[1] # b x h x w    get maximum value along channel
    # idx_y = (idx // w).view(B, 1, 1, h, w).float()
    idx_y = torch.div(idx, w, rounding_mode='floor').view(B, 1, 1, h, w).float()
    idx_x = (idx % w).view(B, 1, 1, h, w).float()
    
    x = torch.linspace(-1, 1, w).to(corr.device)
    y = torch.linspace(-1, 1, h).to(corr.device)

    x = x.view(1,1,w,1,1).expand(B, 1, w, h, w)
    y = y.view(1,h,1,1,1).expand(B, h, 1, h, w)

    gauss_kernel = torch.exp(-((x-idx_x)**2 + (y-idx_y)**2) / (2 * sigma**2))
    gauss_kernel = rearrange(gauss_kernel, 'B h0 w0 h1 w1 -> B (h0 w0) h1 w1')
    # gauss_kernel = gauss_kernel.view(b, head, H, W)
    corr = gauss_kernel * corr
    corr = rearrange(corr, '(b head) hw h1 w1 -> b head hw (h1 w1)', b=b, head=head)

    return corr


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttentionOne(nn.Module):
    """
    Multi-Head Attention module with shared projection
    """
    def __init__(self, n_head=12, d_model=768, d_value=768, dropout=0.1):
        super(MultiHeadAttentionOne, self).__init__()
        self.n_head = n_head
        self.d_k = int(d_model/n_head)
        self.d_v = int(d_value/n_head)

        self.w_q = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_v = nn.Linear(d_value, n_head * self.d_v, bias=False)
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0 / (d_value + self.d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_value)

        self.fc = nn.Linear(n_head * self.d_v, d_value)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, residual=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        output_reshape = True if len(v.shape)==4 else False
        if len(v.shape)==4: 
            bsz, channel, height, weight = v.shape
        if len(q.shape)==4: q = rearrange(q, 'b c h w -> b (h w) c')
        if len(k.shape)==4: k = rearrange(k, 'b c h w -> b (h w) c')
        if len(v.shape)==4: v = rearrange(v, 'b c h w -> b (h w) c')
        if residual is not None:
            if len(residual.shape)==4: residual = rearrange(residual, 'b c h w -> b (h w) c')

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # print(f'q: {q.shape}, k: {k.shape}, v: {v.shape}')
        # print(f'weight shape: {self.w_v.weight.shape}')

        # residual = q
        q = self.w_q(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_k(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_v(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # [(n*b), lq, dk]
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # [(n*b), lk, dk]
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # [(n*b), lv, dv]

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # [b, lq, (n*dv)]

        output = self.dropout(self.fc(output))

        if residual is not None:
            output = self.layer_norm(output + residual)
            # output = output + residual
        else:
            output = self.layer_norm(output)

        if output_reshape:
            output = rearrange(output, 'b (h w) c -> b c h w', h=height, w=weight)
        return output


class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
        pool_ratios=[6,8,10,12]):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t*t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        self.d_convs = nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) for temp in pool_ratios])
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x0, x1):
        # B, N, C = x0.shape
        # H = W = int(N**0.5)
        B, C, H, W = x0.shape
        N = int(H*W)

        if len(x0.shape)==4: x0 = rearrange(x0, 'b c h w -> b (h w) c') 
        # if len(x1.shape)==4: x1 = rearrange(x1, 'b c h w -> b (h w) c') 
        
        q = self.q(x0).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        pools = []
        # x_ = x1.permute(0, 2, 1).reshape(B, C, H, W)
        x_ = x1.clone()
        for (pool_ratio, l) in zip(self.pool_ratios, self.d_convs):
            pool = F.adaptive_avg_pool2d(x_, (round(H/pool_ratio), round(W/pool_ratio)))
            pool = pool + l(pool) # fix backward bug in higher torch versions when training
            pools.append(pool.view(B, C, -1))
        
        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0,2,1))
        
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)   
        x = x.transpose(1,2).contiguous().reshape(B, N, C)
        
        x = self.proj(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x

def save_feature_map(feature, src):
    heatmap = to_pil_image(feature, mode="F")
    cmap = cm.get_cmap("jet")
    overlay = (255 * cmap(np.asarray(heatmap) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray(overlay.astype(np.uint8), 'RGB')
    overlayed_img.save(src)
    # plt.imsave(src, overlay)
    
    # feature = feature.to(torch.device('cpu'))
    # heatmap = feature.mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy()
    # # overlay = cv2.applyColorMap(np.asarray(heatmap), cv2.COLORMAP_JET)
    # overlay = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(src, overlay) 



if __name__=='__main__':
    # query = torch.randn([12, 128, 24, 24])
    # key = value = torch.randn([12, 128, 24, 24])

    # model = MultiHeadAttentionOne(n_head=8, d_model=128, d_value=128)

    # outputs = model(query, key, value)
    # print(outputs.shape)

    # inputs = torch.randn([12, 9216, 128])
    # pool_ratio = [6,8,10,12]
    
    # model = PoolingAttention(dim=128, num_heads=1, pool_ratios=pool_ratio)

    # outputs = model(inputs, inputs)

    # print(outputs.shape)

    query = torch.randn([12, 576, 128])
    key = torch.randn([24, 576, 128])
    value = torch.randn([12, 1152])

    model = MultiHeadedAttention(h=8, d_model=128, add_gaussian=True)

    outputs = model(query, key, value, nshot=2)
    print(outputs.shape)

