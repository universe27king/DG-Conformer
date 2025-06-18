import os

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True
import scipy.signal as signal
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from sklearn import preprocessing
from sklearn import manifold
from convLSTM import ConvLSTM


# writer = SummaryWriter('./TensorBoardX/')
class ChannelAttentionModule(nn.Module):  # 假设输入是多头注意力的输出，即（batchsize, samplelen, feat_num）
    def __init__(self, in_channels, reduction_ratio=4, num_selected_channels=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 32))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
        self.num_selected_channels = num_selected_channels

    def forward(self, input):
        b, _, _ = input.shape
        avg_out = self.avg_pool(input)
        out = self.fc(avg_out)
        out = self.sigmoid(out)  # .view(x.size(0), x.size(1), 1, 1)
        # print(out.shape)            # b,1,32
        topk_out = torch.topk(out, self.num_selected_channels, dim=-1, largest=True)[1]  # b,1,8

        weight_out = out * input
        weight_out = torch.gather(weight_out, 2, topk_out.expand(-1, 200, -1))
        # print(weight_out.shape)          # 2, 200, 8

        return weight_out



class phase_embedding(nn.Module):
    def __init__(self, phase_info, time_steps):
        super(phase_embedding, self).__init__()

        time_steps = np.arange(time_steps)
        sinterm = np.sin(phase_info * time_steps)
        costerm = np.cos(phase_info * time_steps)
        self.phaseembd = (sinterm + costerm) / 2

    def forward(self, inputdata):                         # inputdata格式为240 * 8 * 200
        output = inputdata + self.phaseembd
        return output


class SamePadConv_timedim(nn.Module):                     # 计算时间维度same padding
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups):
        super(SamePadConv_timedim, self).__init__()
        padding_row = (kernel_size[1] - 1) // 2
        padding_column = (kernel_size[0] - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(padding_column, padding_row), groups=groups)
    def forward(self, x):
        return self.conv(x)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    # def __init__(self, max_len, d_model, dropout):
    def __init__(self, max_len, d_model):

        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=64):
        # self.patch_size = patch_size
        super().__init__()


        # 实验加入convLSTM
        self.ConvLSTM = ConvLSTM(input_dim=1, hidden_dim=[32], kernel_size=(1, 3), num_layers=1, batch_first=True,
                     bias=True, return_all_layers=False)


        self.globalnet = nn.Sequential(            # 一维转变成二维
            nn.Conv2d(32, emb_size, kernel_size=(9, 1), stride=(9, 1), groups=1, bias=False),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),



            # nn.Conv2d(1, emb_size, kernel_size=(9, 1), stride=(9, 1), groups=1, bias=False),
            # nn.BatchNorm2d(emb_size),
            # nn.ReLU()
        )

        self.projection = nn.Sequential(                     # 231101目前来看如果不加transformer的话这里好像有点多余
            # nn.Conv2d(128, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            # nn.BatchNorm2d(emb_size),
            # nn.ReLU(),

            Rearrange('b e (h) (w) -> b (h w) e')
        )

        self.positionalencoding = PositionalEncoding(max_len=500, d_model=emb_size)




    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape

        # 实验加入ConvLSTM
        x = x.unsqueeze(1)
        _, last_states = self.ConvLSTM(x)  # last_states是list类型
        x = last_states[0][0]


        x = self.globalnet(x)
        x = self.projection(x)
        x = self.positionalencoding(x)
        return x


# 简单测试：conformer里只放这个simple_net
class simple_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.timeconv = nn.Sequential(            # 一维转变成二维
            SamePadConv_timedim(1, 32, kernel_size=(1, 125), stride=(1, 1)),             # padding是默认两边都填？能否只在右边？
            nn.ReLU(),
            nn.BatchNorm2d(32))
        self.electrodconv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(8, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.AvgPool2d(kernel_size=(1, 3))
        )
        self.projection = nn.Sequential(                     # 231101目前来看如果不加transformer的话这里好像有点多余
            Rearrange('b e (h) (w) -> b (e h w)'),
            nn.Linear(64 * 66, 40)
        )

    def forward(self, x):
        timeconv = self.timeconv(x)
        # print(timeconv.shape)
        residual_connect = x + timeconv
        electrodconv = self.electrodconv(residual_connect)
        # b, feat_dim, elec_dim, sample_len = electrodconv.shape
        # print(electrodconv.shape)
        out = self.projection(electrodconv)
        return out



class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):       # **kwargs用法:预先不知道函数使用者会传递多少个参数, 将一个可变的关键字参数的字典传给函数实参
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        # print(queries.shape)        # torch.Size([64, 8, 25, 5])
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        # print(energy.shape)           # torch.Size([32, 8, 25, 25])

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)

        # print(att.shape)         # 大多数是0.005周围的，因为1/200=0.005
        # att中<0.3的置零,否则不变
        # att = torch.where(att < 0.0045, 0, att)


        # 注意drop和mask的区别——一个对网络节点权重丢弃，一个对特征数据丢弃
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out




class spatialreduction_MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        # print(queries.shape)        # torch.Size([64, 8, 25, 5])
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        # print(energy.shape)           # torch.Size([32, 8, 25, 25])

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)

        # print(att.shape)         # 大多数是0.005周围的，因为1/200=0.005
        # att中<0.3的置零,否则不变
        # att = torch.where(att < 0.0045, 0, att)


        # 注意drop和mask的区别——一个对网络节点权重丢弃，一个对特征数据丢弃
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class Attention_withpyramid(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=(1, sr_ratio), stride=(1, sr_ratio))        # B, C, H, W -> B, C, H // sr_ratio, W // sr_ratio
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        H, W = 1, 200 // self.sr_ratio
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)              # B, N, C -> B, C, N -> B, C, H, W
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)      # B, C, H // sr_ratio, W // sr_ratio -> B, C, N_new -> B, N_new, C
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)   # B, N_new, C -> B, N_new, C*2 -> B, C_new, 2, num_heads, C // self.num_heads -> ...
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]        # B, num_heads, C_new, C // self.num_heads

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x





class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 # num_heads=10,
                 num_heads=8,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(

            # 无效的金字塔尝试
            # nn.Sequential(
            #     # nn.LayerNorm(emb_size),                 # 似乎删去这个LN效果可以提升一点点
            #     Attention_withpyramid(emb_size, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1),
            #     # MultiHeadAttention(emb_size, num_heads, drop_p),
            #     nn.Dropout(drop_p)
            # ),
            # nn.Sequential(
            #     # nn.LayerNorm(emb_size),                 # 似乎删去这个LN效果可以提升一点点
            #     Attention_withpyramid(emb_size, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2),
            #     # MultiHeadAttention(emb_size, num_heads, drop_p),
            #     nn.Dropout(drop_p)
            # ),
            # nn.Sequential(
            #     # nn.LayerNorm(emb_size),                 # 似乎删去这个LN效果可以提升一点点
            #     Attention_withpyramid(emb_size, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=3),
            #     # MultiHeadAttention(emb_size, num_heads, drop_p),
            #     nn.Dropout(drop_p)
            # ),


            ResidualAdd(nn.Sequential(
                # nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            # ResidualAdd(nn.Sequential(
            #     nn.LayerNorm(emb_size),
            #     FeedForwardBlock(
            #         emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
            #     nn.Dropout(drop_p)
            # ))
        )
        # self.firstAtt = nn.Sequential(
        #     # nn.LayerNorm(emb_size),                 # 似乎删去这个LN效果可以提升一点点
        #     spatialreduction_MultiHeadAttention(emb_size, num_heads, drop_p),
        #     # MultiHeadAttention(emb_size, num_heads, drop_p),
        #     nn.Dropout(drop_p)
        # )
        # self.secAtt = nn.Sequential(
        #     # nn.LayerNorm(emb_size),                 # 似乎删去这个LN效果可以提升一点点
        #     spatialreduction_MultiHeadAttention(emb_size, num_heads, drop_p),
        #     # MultiHeadAttention(emb_size, num_heads, drop_p),
        #     nn.Dropout(drop_p)
        # )
        #
        # def forward(input):
        #     res = self.firstAtt(input)
        #     input
        #     return res

class global_cat_local(nn.Module):
    def __init__(self, depth=1, emb_size=64):
        super(global_cat_local, self).__init__()
        self.global_feat = nn.Sequential(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size)
        )
        self.regional_feat = nn.Sequential(
            SamePadConv_timedim(1, 8, kernel_size=(1, 25), stride=(1, 1), groups=1),        # stride不等于1的时候似乎无法same padding
            nn.BatchNorm2d(8),
            nn.ELU(),
            # 深度可分离卷积
            nn.Conv2d(8, 8, kernel_size=(9, 1), stride=(9, 1), groups=8),
            nn.Conv2d(8, emb_size, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout(0.5),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.deepresshrink = Shrinkage(emb_size, gap_size=(1))

    def forward(self, input):
        global_feature = self.global_feat(input)
        # global_feature = self.deepresshrink(global_feature)

        regional_feature = self.regional_feat(input)
        # 特征融合用直接相加还是维度拼接
        feat_fusion = global_feature + regional_feature
        return feat_fusion   #, global_feature



# class TransformerEncoder(nn.Sequential):
#     def __init__(self, depth, emb_size):
#         super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

# 实验金字塔结构transformer块
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(
            TransformerEncoderBlock(emb_size)

        )


class WeightFreezing(nn.Module):
    def __init__(self, input_dim, output_dim, shared_ratio=0.3, multiple=0):
        super(WeightFreezing, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        mask = torch.rand(input_dim, output_dim) < shared_ratio
        self.register_buffer('shared_mask', mask)
        self.register_buffer('independent_mask', ~mask)

        self.multiple = multiple

    def forward(self, x, shared_weight):
        combined_weight = torch.where(self.shared_mask, shared_weight*self.multiple, self.weight.t())      # shared_mask为True的位置返回shared_weight*self.multiple，为False的位置返回weight.t()
        output = F.linear(x, combined_weight.t(), self.bias)
        return output



class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        # global average pooling
        # self.clshead = nn.Sequential(
        #     Reduce('b n e -> b e', reduction='mean'),
        #     nn.LayerNorm(emb_size),
        #     nn.Linear(emb_size, n_classes)
        # )
        self.fc = nn.Sequential(
            nn.Conv2d(emb_size, 8, kernel_size=(1, 1), stride=(1, 1)),     # 突然想到直接用linear层（32，8）也可以实现。而且cnn谁知道切分的时候是怎么搞得呢
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d((1, 2), (1, 2)),           # 实验发现1*1卷积结合kernel_size=(1, 2)pool效果比单独kernel_size=(1, 2)卷积效果要好
            Rearrange('b (e) (m) (o)-> b (e m o)'),
            nn.Dropout(0.5),
            nn.Linear(8 * 100, n_classes)              # 输入数据为200points时, 对应输入长度全连接层维度对应要改为(224, ...); 250points时304
        )

        self.feedforward = ResidualAdd(nn.Sequential(
            # nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Dropout(0.5)
            ))
        # self.chanmean = nn.AvgPool2d((1, emb_size), (1, emb_size))
        self.chanmean = nn.Sequential(
            nn.Linear(emb_size, 32),
            nn.GELU(),
            #为换tok把全连接层在这里截断
            nn.Linear(32, 4),
            nn.Dropout(0.7)
            )
        # self.timemean = nn.AvgPool2d((1, 2), (1, 2))
        self.lastlinear = nn.Sequential(
        #     # nn.Dropout(0.75),      # 为了实验一下R-drop方法，将最后的分类层前面的drop移到取tok之前
             nn.Linear(200 * 4, 40)
             )

        # 冻结分类层权重
        #shared_ratio = 0.1
        #self.classifier = WeightFreezing(800, 40, shared_ratio=shared_ratio)
        #self.shared_weights = nn.Parameter(torch.Tensor(40, 800), requires_grad=False)
        #self.bias = nn.Parameter(torch.Tensor(40))
        #nn.init.kaiming_uniform_(self.shared_weights, a=math.sqrt(5))
        #if self.bias is not None:
        #    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.shared_weights)
        #    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #    nn.init.uniform_(self.bias, -bound, bound)
        #self.fixed_weight = self.shared_weights.t() * self.classifier.shared_mask


    def forward(self, x):
        # x = x.contiguous().view(x.size(0), -1)

        # 231108添加SE_net
        # x = self.feat_select(x)

        x = self.feedforward(x)
        # x = repeat(x, 'b (o) e -> b e (m) (o)', m=1)
        x = self.chanmean(x)
        tok = rearrange(x, 'b o e -> b e o')
        # x = self.timemean(x)
        x = rearrange(tok, 'b e o -> b (e o)')
        # out = self.fc(x)
        out = self.lastlinear(x)
        # out = self.classifier(x, self.fixed_weight.to(x.device))
        return tok, out


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(4 * 200, 40)

    def forward(self, features):
        return self.fc(features)



class Conformer(nn.Sequential):
    def __init__(self, emb_size=64, depth=1, n_classes=40, **kwargs):      # EEG conformer源码使用的是depth=6个叠加, 怀疑可能造成过拟合, 参考TFF源码用的是一个
        super().__init__(
            # # 最原始源代码中此类中为PatchEmbedding(emb_size),TransformerEncoder(depth, emb_size),ClassificationHead(emb_size, n_classes)三部分
            # PatchEmbedding(emb_size),
            # TransformerEncoder(depth, emb_size),
            # ClassificationHead(emb_size, n_classes)

            global_cat_local(depth, emb_size),
            ClassificationHead(emb_size, n_classes)

            # 实验时尝试过的全CNN简单网络和参考文献EEGnet
            # simple_transformer(emb_size, emb_size, num_heads=4),
            # EEGnet_SSVEP_compact()
        )
        self.register_buffer('pre_features', torch.zeros(64, 200))  # 参数含义见上例句
        self.register_buffer('pre_weight1', torch.ones(64, 1))
        




# 231225实验软阈值化
class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1], 1)    # B,200,64->B,64,200,1
        x_raw = x
        x = torch.abs(x)   # 为什么要取绝对值？
        x_abs = x
        x = self.gap(x)     # 源码tf格式用了两个keep_dims=True的reduce_mean
        x = torch.flatten(x, 1)
        average = torch.mean(x, dim=1, keepdim=True)  # CS
        # average = x    # CW
        # print(x.shape)   # 64, 64
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.expand(x.shape[0], x.shape[1], x_raw.shape[2], 1)


        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        # print(x.shape)   # torch.Size([64, 64, 200, 1])
        x = x.squeeze(3).permute(0, 2, 1)
        print(x.shape)
        return x





class LabelSmoothingLoss(nn.Module):
    "Implement label smoothing."

    def __init__(self, class_num=40, smoothing=0.01):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.class_num = class_num

    def forward(self, x, target):
        assert x.size(1) == self.class_num
        if self.smoothing == None:
            return nn.CrossEntropyLoss()(x, target)

        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.class_num - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)  # 独热

        logprobs = F.log_softmax(x, dim=-1)
        mean_loss = -torch.sum(true_dist * logprobs) / x.size(-2)
        return mean_loss






class EEGnet_SSVEP_compact(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayer = nn.Sequential(
            # pytorch实现keras的DepthwiseConv2D和SeparableConv2D
            # depthwise conv: groups设为in_channels，同时out_channels也设为与in_channels相同
            SamePadConv_timedim(1, 96, kernel_size=(1, 200), stride=(1, 1), groups=1),               # 因为卷积核是偶数，所以导致same padding之后的conv输出尺寸比输入少1
            nn.BatchNorm2d(96),
            # 下面一行为深度卷积
            nn.Conv2d(96, 96, kernel_size=(8, 1), groups=96),
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5),
            # 下面两行为深度可分离卷积
            SamePadConv_timedim(96, 96, kernel_size=(1, 16), stride=(1, 1), groups=96),
            nn.Conv2d(96, 96, kernel_size=(1, 1)),

            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5),
        )

        self.classification = nn.Sequential(
            nn.Linear(576, 40),
            nn.Softmax()
        )

    def forward(self, input):
        b = input.shape[0]
        conv = self.convlayer(input)
        conv = conv.view(b, -1)
        out = self.classification(conv)
        return out






