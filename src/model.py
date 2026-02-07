from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

# new
class LinearBlock(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(dim, int(expansion_factor * dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(expansion_factor * dim), dim),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        # 残差连接：x + 变换(归一化(x))
        return x + self.fn(self.ln(x))

class TextProj(nn.Module):
    def __init__(self, embedding_dim=4096, output_dim=512, num_layers=1): # 这里把层数改小，去掉layernorm，防止投影后趋同
        super().__init__()
        # 4层残差块，保持 4096 维度进行非线性映射
        self.text_adaptor = nn.Sequential(
            *[LinearBlock(embedding_dim) for _ in range(num_layers)],
            # nn.LayerNorm(embedding_dim),
            # 最终投影到与视觉特征相同的维度 (如 512 或 1024)
            nn.Linear(embedding_dim, output_dim)
        )
    
    def forward(self, text_emb):
        # 输入 text_emb: [Batch, 4096]
        # L2 归一化是对比学习的标准操作，确保特征在同一球面上
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        return self.text_adaptor(text_emb)
    
class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))

        self.text_proj = TextProj(embedding_dim=4096, output_dim=visual_width) # new

        self.classifier = nn.Linear(visual_width, 1)

        # self.clipmodel, _ = clip.load("ViT-B/16", device)
        # for clip_param in self.clipmodel.parameters():
        #     clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        # self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        # nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features

    def forward(self, visual, padding_mask, text_features_llm, lengths):
        """
        visual: [Batch, T, D] 视觉特征
        text_features_llm: [Batch, 4096] 文本特征
        """
        visual_features = self.encode_video(visual, padding_mask, lengths) # Fv

        # logits1: [B, T, 1]
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        # 视觉特征全局池化
        # 利用 logits1 (即时间注意力分数) 对时序特征进行加权求和
        # 得分越高的片段（越可能是异常的片段）对全局视频特征贡献越大
        weights = torch.sigmoid(logits1) # [B, T, 1] 归一化到 0-1
        # V_global: [B, D] 代表整段视频的全局视觉向量
        v_global = torch.sum(visual_features * weights, dim=1) / (torch.sum(weights, dim=1) + 1e-6)

        # 4. 文本特征投影
        # 将 4096 维映射到视觉维度 D
        # t_global: [B, D]
        t_global = self.text_proj(text_features_llm)

        # ==================== 视觉注入逻辑 ====================


        # 归一化视觉池化向量
        v_significant_norm = F.normalize(v_global, p=2, dim=-1)

        t_global_injected = t_global + v_significant_norm
        t_global_refined = t_global_injected + self.mlp1(t_global_injected)

        # 5. 特征对齐与归一化
        # 按照论文要求，进行 1:1 的直接对齐
        v_norm = v_significant_norm
        t_norm = F.normalize(t_global_refined, p=2, dim=-1)

        # text_features_ori = self.encode_textprompt(text)

        # text_features = text_features_ori
        # logits_attn = logits1.permute(0, 2, 1)
        # visual_attn = logits_attn @ visual_features
        # visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        # visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        # text_features = text_features_ori.unsqueeze(0)
        # text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        # text_features = text_features + visual_attn
        # text_features = text_features + self.mlp1(text_features)

        # visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        # text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        # text_features_norm = text_features_norm.permute(0, 2, 1)
        # logits2 = (v_norm @ t_norm.T) / 0.07 
        logit_scale = self.logit_scale.exp()
        logits2 = (v_norm @ t_norm.T) * logit_scale

        # logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        return visual_features, t_norm, logits1, logits2
    