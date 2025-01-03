# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple, Type, Optional

import torch
from torch import Tensor, nn

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        """
        max_seq_len: 最大支持的位置长度
        d_model: 位置编码的维度（通常与模型的嵌入维度一致）
        """
        super(SinusoidalPositionEncoding, self).__init__()
        self.d_model = d_model

        # 生成正弦/余弦位置编码矩阵
        position = torch.arange(0, max_seq_len).unsqueeze(1)  # [max_seq_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )  # [d_model/2]

        # 偶数维度为 sin，奇数维度为 cos
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        # 注册位置编码矩阵，不作为模型参数
        self.register_buffer('pe', pe)  # [max_seq_len, d_model]

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        position_ids: 位置索引张量 [batch_size, seq_len] 或 [seq_len]
        返回: 对应位置的编码张量 [batch_size, seq_len, d_model] 或 [seq_len, d_model]
        """
        return self.pe[position_ids]  # 根据 position_ids 索引位置编码

class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        max_position_emb: int = 4096,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    max_position_emb=max_position_emb,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, 
            max_position_emb=max_position_emb
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
        #create pos emb
        self.imag_pos_layer = SinusoidalPositionEncoding(max_seq_len=max_position_emb, d_model=embedding_dim)
        

    def forward(
        self,
        vlm_embedding: Tensor,
        vlm_position_ids: Tensor,
        seg_embedding: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          vlm_embedding (torch.Tensor): image to attend to. Should be shape
            B x K x embedding_dim.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as vlm_embedding.
          seg_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed seg_embedding
          torch.Tensor: the processed vlm_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, k, emb_dim = vlm_embedding.shape
        vlm_pe = self.imag_pos_layer(vlm_position_ids)
        assert vlm_pe.shape[1] == vlm_embedding.shape[1]

        # Prepare queries
        queries = seg_embedding
        keys = vlm_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=seg_embedding,
                key_pe=vlm_pe,
                attention_mask=attention_mask,
            )

        # Apply the final attention layer from the points to the image
        q = queries + seg_embedding
        k = keys + vlm_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys, attention_mask=attention_mask)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        max_position_emb: int = 4096,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of seg
        inputs, (2) cross attention of seg inputs to vlm inputs, (3) mlp
        block on seg inputs, and (4) cross attention of vlm inputs to seg
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads,)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
            max_position_emb=max_position_emb
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, 
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        #query是seg， key是vlm
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        # 这里是seg 关注img 需要mask来防止关注padding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys, 
                                                  attention_mask=attention_mask)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        max_position_emb: int = 4096,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attention_mask=None) -> Tensor:
        
        q_length, k_length = q.shape[1], k.shape[1]
        try:
            if q_length < k_length:
                assert attention_mask is not None
            else:
                assert attention_mask is None
        except:
            from IPython import embed; embed(); exit()
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens

        attn = attn / math.sqrt(c_per_head)
        #--加入attention-mask, attention_mask.shape:[N_tokens, N_tokens]
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].to(q.dtype)
            attn += (1 - attention_mask) * -1e9
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

if __name__ == '__main__':
    # 示例
    max_seq_len = 100  # 最大支持的序列长度
    d_model = 64       # 位置编码的维度
    position_encoding = SinusoidalPositionEncoding(max_seq_len, d_model)

    # 输入 position_ids
    position_ids = torch.tensor([[0, 1, 2], [0, 4, 0]])  # [batch_size, seq_len]
    output = position_encoding(position_ids)

    print(output.shape)  # 输出: [2, 3, 64]
    print(output[0, 0])  # 打印位置 0 的编码
    print(output[0, 1])  # 打印位置 0 的编码
    print(output[1, 0])  # 打印位置 0 的编码
    print(output[1, -1])  # 打印位置 0 的编码