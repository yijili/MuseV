# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
import logging

from einops import rearrange

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.transformer_2d import (
    Transformer2DModelOutput,
    Transformer2DModel as DiffusersTransformer2DModel,
)

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import ImagePositionalEmbeddings
from diffusers.utils import BaseOutput, deprecate
from diffusers.models.attention import (
    BasicTransformerBlock as DiffusersBasicTransformerBlock,
)
from diffusers.models.embeddings import PatchEmbed
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.constants import USE_PEFT_BACKEND

from .attention import BasicTransformerBlock

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.info)

# 本部分 与 diffusers/models/transformer_2d.py 几乎一样
# 更新部分
# 1. 替换自定义 BasicTransformerBlock 类
# 2. 在forward 里增加了 self_attn_block_embs 用于 提取 self_attn 中的emb

# this module is same as diffusers/models/transformer_2d.py. The update part is
# 1 redefine BasicTransformerBlock
# 2. add self_attn_block_embs in forward to extract emb from self_attn


class Transformer2DModel(DiffusersTransformer2DModel):
    """
    用于图像类数据的2D Transformer模型。

    参数:
        num_attention_heads (`int`, *可选*, 默认为 16): 多头注意力机制中使用的头数。
        attention_head_dim (`int`, *可选*, 默认为 88): 每个注意力头的通道数。
        in_channels (`int`, *可选*):
            输入和输出的通道数（如果输入是**连续的**）。
        num_layers (`int`, *可选*, 默认为 1): Transformer块的层数。
        dropout (`float`, *可选*, 默认为 0.0): 使用的dropout概率。
        cross_attention_dim (`int`, *可选*): `encoder_hidden_states`维度数。
        sample_size (`int`, *可选*): 潜在图像的宽度（如果输入是**离散的**）。
            在训练期间是固定的，因为它用于学习位置嵌入的数量。
        num_vector_embeds (`int`, *可选*):
            潜在像素向量嵌入的类别数（如果输入是**离散的**）。
            包括被遮罩的潜在像素类别。
        activation_fn (`str`, *可选*, 默认为 `"geglu"`): 前馈网络中使用的激活函数。
        num_embeds_ada_norm ( `int`, *可选*):
            训练期间使用的扩散步骤数。如果至少一个归一化层是
            `AdaLayerNorm`，则需要传递此参数。在训练期间是固定的，因为它用于学习添加到隐藏状态的嵌入数量。

            在推理过程中，您可以去噪的步数不超过但不能多于`num_embeds_ada_norm`。
        attention_bias (`bool`, *可选*):
            配置`TransformerBlocks`注意力是否应包含偏置参数。
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int | None = None,
        out_channels: int | None = None,
        num_layers: int = 1,
        dropout: float = 0,
        norm_num_groups: int = 32,
        cross_attention_dim: int | None = None,
        attention_bias: bool = False,
        sample_size: int | None = None,
        num_vector_embeds: int | None = None,
        patch_size: int | None = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int | None = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        attention_type: str = "default",
        cross_attn_temporal_cond: bool = False,
        ip_adapter_cross_attn: bool = False,
        need_t2i_facein: bool = False,
        need_t2i_ip_adapter_face: bool = False,
        image_scale: float = 1.0,
    ):
        """
        初始化Transformer2DModel
        
        参数说明:
            num_attention_heads: 注意力头的数量
            attention_head_dim: 每个注意力头的维度
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_layers: Transformer块的层数
            dropout: Dropout比率
            norm_num_groups: 归一化组数
            cross_attention_dim: 交叉注意力的维度
            attention_bias: 是否使用注意力偏置
            sample_size: 样本大小
            num_vector_embeds: 向量嵌入的数量
            patch_size: 补丁大小
            activation_fn: 激活函数类型
            num_embeds_ada_norm: AdaLayerNorm的嵌入数量
            use_linear_projection: 是否使用线性投影
            only_cross_attention: 是否仅使用交叉注意力
            double_self_attention: 是否使用双重自注意力
            upcast_attention: 是否上转换注意力
            norm_type: 归一化类型
            norm_elementwise_affine: 是否元素级仿射变换
            attention_type: 注意力类型
            cross_attn_temporal_cond: 是否使用交叉注意力时间条件
            ip_adapter_cross_attn: 是否使用IP适配器交叉注意力
            need_t2i_facein: 是否需要T2I FaceIn
            need_t2i_ip_adapter_face: 是否需要T2I IP适配器Face
            image_scale: 图像缩放因子
        """
        logger.debug("初始化Transformer2DModel")
        super().__init__(
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            num_layers,
            dropout,
            norm_num_groups,
            cross_attention_dim,
            attention_bias,
            sample_size,
            num_vector_embeds,
            patch_size,
            activation_fn,
            num_embeds_ada_norm,
            use_linear_projection,
            only_cross_attention,
            double_self_attention,
            upcast_attention,
            norm_type,
            norm_elementwise_affine,
            attention_type,
        )
        inner_dim = num_attention_heads * attention_head_dim
        logger.info(f"创建{num_layers}个BasicTransformerBlock")
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    attention_type=attention_type,
                    cross_attn_temporal_cond=cross_attn_temporal_cond,
                    ip_adapter_cross_attn=ip_adapter_cross_attn,
                    need_t2i_facein=need_t2i_facein,
                    need_t2i_ip_adapter_face=need_t2i_ip_adapter_face,
                    image_scale=image_scale,
                )
                for d in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.cross_attn_temporal_cond = cross_attn_temporal_cond
        self.ip_adapter_cross_attn = ip_adapter_cross_attn

        self.need_t2i_facein = need_t2i_facein
        self.need_t2i_ip_adapter_face = need_t2i_ip_adapter_face
        self.image_scale = image_scale
        self.print_idx = 0
        logger.debug("Transformer2DModel初始化完成")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        self_attn_block_embs: Optional[List[torch.Tensor]] = None,
        self_attn_block_embs_mode: Literal["read", "write"] = "write",
        return_dict: bool = True,
    ):
        """
        Transformer2DModel 的前向传播方法。

        参数:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                输入 `hidden_states`。
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                交叉注意力层的条件嵌入。如果未提供，则交叉注意力默认为自注意力。
            timestep ( `torch.LongTensor`, *optional*):
                用于指示去噪步骤。可选的时间步长作为嵌入应用于 `AdaLayerNorm`。
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                用于指示类别标签条件。可选的类别标签作为嵌入应用于 `AdaLayerZeroNorm`。
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                如果指定，则传递给 `AttentionProcessor` 的 kwargs 字典，定义在
                diffusers.models.attention_processor https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py 中的 `self.processor` 下。
            attention_mask ( `torch.Tensor`, *optional*):
                形状为 `(batch, key_tokens)` 的注意力掩码应用于 `encoder_hidden_states`。如果为 `1` 则保留掩码，
                否则如果为 `0` 则丢弃。掩码将被转换为偏置，这会将大的负值添加到与"丢弃"标记对应的注意力分数中。
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                应用于 `encoder_hidden_states` 的交叉注意力掩码。支持两种格式：

                    * 掩码 `(batch, sequence_length)` True = 保留, False = 丢弃。
                    * 偏置 `(batch, 1, sequence_length)` 0 = 保留, -10000 = 丢弃。

                如果 `ndim == 2`: 将被解释为掩码，然后转换为与上述格式一致的偏置。此偏置将被添加到交叉注意力分数中。
            return_dict (`bool`, *optional*, 默认为 `True`):
                是否返回 [`~models.unet_2d_condition.UNet2DConditionOutput`] 而不是普通元组。

        返回:
            如果 `return_dict` 为 True，则返回 [`~models.transformer_2d.Transformer2DModelOutput`]，否则返回
            第一个元素为样本张量的 `tuple`。
        """
        logger.debug(f"Transformer2DModel前向传播开始，输入hidden_states形状: {hidden_states.shape}")
        
        # 确保 attention_mask 是偏置，并给它一个单例 query_tokens 维度。
        #   我们可能已经完成了这个转换，例如如果我们通过 UNet2DConditionModel#forward 到达这里。
        #   我们可以通过计算维度来判断；如果 ndim == 2: 它是掩码而不是偏置。
        # 期望的掩码形状：
        #   [batch, key_tokens]
        # 添加单例 query_tokens 维度：
        #   [batch,                    1, key_tokens]
        # 这有助于将其作为偏置广播到注意力分数上，注意力分数将采用以下形状之一：
        #   [batch,  heads, query_tokens, key_tokens] (例如 torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (例如 xformers 或经典 attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # 假设掩码表示为：
            #   (1 = 保留,      0 = 丢弃)
            # 将掩码转换为可以添加到注意力分数的偏置：
            #       (保留 = +0,     丢弃 = -10000.0)
            logger.debug("处理attention_mask")
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 以与处理 attention_mask 相同的方式将 encoder_attention_mask 转换为偏置
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            logger.debug("处理encoder_attention_mask")
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 检索 lora 缩放因子。
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )
        logger.debug(f"获取lora_scale: {lora_scale}")

        # 1. 输入处理
        if self.is_input_continuous:
            logger.debug("处理连续输入")
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                logger.debug("使用非线性投影")
                hidden_states = (
                    self.proj_in(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_in(hidden_states)
                )
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                    batch, height * width, inner_dim
                )
            else:
                logger.debug("使用线性投影")
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                    batch, height * width, inner_dim
                )
                hidden_states = (
                    self.proj_in(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_in(hidden_states)
                )

        elif self.is_input_vectorized:
            logger.debug("处理向量化输入")
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            logger.debug("处理补丁输入")
            height, width = (
                hidden_states.shape[-2] // self.patch_size,
                hidden_states.shape[-1] // self.patch_size,
            )
            hidden_states = self.pos_embed(hidden_states)

            if self.adaln_single is not None:
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                    )
                batch_size = hidden_states.shape[0]
                timestep, embedded_timestep = self.adaln_single(
                    timestep,
                    added_cond_kwargs,
                    batch_size=batch_size,
                    hidden_dtype=hidden_states.dtype,
                )

        # 2. 块处理
        if self.caption_projection is not None:
            logger.debug("处理caption_projection")
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        logger.debug(f"开始处理{len(self.transformer_blocks)}个transformer块")
        for i, block in enumerate(self.transformer_blocks):
            logger.debug(f"处理第{i}个transformer块")
            if self.training and self.gradient_checkpointing:
                logger.debug("使用梯度检查点")
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    self_attn_block_embs,
                    self_attn_block_embs_mode,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    self_attn_block_embs=self_attn_block_embs,
                    self_attn_block_embs_mode=self_attn_block_embs_mode,
                )
            # 将 转换 self_attn_emb的尺寸
            if (
                self_attn_block_embs is not None
                and self_attn_block_embs_mode.lower() == "write"
            ):
                self_attn_idx = block.spatial_self_attn_idx
                if self.print_idx == 0:
                    logger.debug(
                        f"self_attn_block_embs, num={len(self_attn_block_embs)}, before, shape={self_attn_block_embs[self_attn_idx].shape}, height={height}, width={width}"
                    )
                self_attn_block_embs[self_attn_idx] = rearrange(
                    self_attn_block_embs[self_attn_idx],
                    "bt (h w) c->bt c h w",
                    h=height,
                    w=width,
                )
                if self.print_idx == 0:
                    logger.debug(
                        f"self_attn_block_embs, num={len(self_attn_block_embs)},  after ,shape={self_attn_block_embs[self_attn_idx].shape}, height={height}, width={width}"
                    )

        if self.proj_out is None:
            logger.debug("proj_out为None，直接返回hidden_states")
            return hidden_states

        # 3. 输出处理
        if self.is_input_continuous:
            logger.debug("处理连续输出")
            if not self.use_linear_projection:
                hidden_states = (
                    hidden_states.reshape(batch, height, width, inner_dim)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                hidden_states = (
                    self.proj_out(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_out(hidden_states)
                )
            else:
                hidden_states = (
                    self.proj_out(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_out(hidden_states)
                )
                hidden_states = (
                    hidden_states.reshape(batch, height, width, inner_dim)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )

            output = hidden_states + residual
        elif self.is_input_vectorized:
            logger.debug("处理向量化输出")
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()

        if self.is_input_patches:
            logger.debug("处理补丁输出")
            if self.config.norm_type != "ada_norm_single":
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = (
                    self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                )
                hidden_states = self.proj_out_2(hidden_states)
            elif self.config.norm_type == "ada_norm_single":
                shift, scale = (
                    self.scale_shift_table[None] + embedded_timestep[:, None]
                ).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.squeeze(1)

            # unpatchify
            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(
                    -1,
                    height,
                    width,
                    self.patch_size,
                    self.patch_size,
                    self.out_channels,
                )
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(
                    -1,
                    self.out_channels,
                    height * self.patch_size,
                    width * self.patch_size,
                )
            )
        self.print_idx += 1
        logger.debug(f"前向传播完成，输出形状: {output.shape}")
        if not return_dict:
            logger.debug("以元组形式返回结果")
            return (output,)

        logger.debug("以Transformer2DModelOutput形式返回结果")
        return Transformer2DModelOutput(sample=output)