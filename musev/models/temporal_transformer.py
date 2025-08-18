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

# Adapted from https://github.com/huggingface/diffusers/blob/v0.16.1/src/diffusers/models/transformer_temporal.py
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Literal, Optional
import logging

import torch
from torch import nn
from einops import rearrange, repeat

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformer_temporal import (
    TransformerTemporalModelOutput,
    TransformerTemporalModel as DiffusersTransformerTemporalModel,
)
from diffusers.models.attention_processor import AttnProcessor

from mmcm.utils.gpu_util import get_gpu_status
from ..data.data_util import (
    batch_concat_two_tensor_with_index,
    batch_index_fill,
    batch_index_select,
    concat_two_tensor,
    align_repeat_tensor_single_dim,
)
from ..utils.attention_util import generate_sparse_causcal_attn_mask
from .attention import BasicTransformerBlock
from .attention_processor import (
    BaseIPAttnProcessor,
)
from . import Model_Register

# https://github.com/facebookresearch/xformers/issues/845
# 输入bs*n_frames*w*h太高，xformers报错。因此将transformer_temporal的allow_xformers均关掉
# if bs*n_frames*w*h to large, xformers will raise error. So we close the allow_xformers in transformer_temporal
logger = logging.getLogger(__name__)


@Model_Register.register
class TransformerTemporalModel(ModelMixin, ConfigMixin):
    """
    用于视频类数据的Transformer模型。

    参数:
        num_attention_heads (`int`, *optional*, 默认为 16): 多头注意力机制中头的数量。
        attention_head_dim (`int`, *optional*, 默认为 88): 每个注意力头的通道数。
        in_channels (`int`, *optional*):
            如果输入是连续的，请传递此参数。表示输入和输出的通道数。
        num_layers (`int`, *optional*, 默认为 1): 要使用的Transformer块的层数。
        dropout (`float`, *optional*, 默认为 0.0): 使用的dropout概率。
        cross_attention_dim (`int`, *optional*): encoder_hidden_states的维度数。
        sample_size (`int`, *optional*): 如果输入是离散的，请传递此参数。表示潜在图像的宽度。
            注意，这在训练时是固定的，因为它用于学习位置嵌入的数量。参见
            `ImagePositionalEmbeddings`。
        activation_fn (`str`, *optional*, 默认为 `"geglu"`): 前馈网络中使用的激活函数。
        attention_bias (`bool`, *optional*):
            配置TransformerBlocks的注意力是否应包含偏置参数。
        double_self_attention (`bool`, *optional*):
            配置每个TransformerBlock是否应包含两个自注意力层
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        femb_channels: Optional[int] = None,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
        allow_xformers: bool = False,
        only_cross_attention: bool = False,
        keep_content_condition: bool = False,
        need_spatial_position_emb: bool = False,
        need_temporal_weight: bool = True,
        self_attn_mask: str = None,
        # TODO: 运行参数，有待改到forward里面去
        # TODO: running parameters, need to be moved to forward
        image_scale: float = 1.0,
        processor: AttnProcessor | None = None,
        remove_femb_non_linear: bool = False,
    ):
        """
        初始化TransformerTemporalModel模型
        
        Args:
            num_attention_heads: 注意力头数量
            attention_head_dim: 每个注意力头的维度
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_layers: Transformer块层数
            femb_channels: 帧嵌入通道数
            dropout: Dropout概率
            norm_num_groups: GroupNorm中的组数
            cross_attention_dim: 交叉注意力维度
            attention_bias: 是否在注意力中使用偏置
            sample_size: 样本大小
            activation_fn: 激活函数类型
            norm_elementwise_affine: 是否在归一化中使用可学习参数
            double_self_attention: 是否使用双重自注意力
            allow_xformers: 是否允许使用xformers优化
            only_cross_attention: 是否仅使用交叉注意力
            keep_content_condition: 是否保持内容条件
            need_spatial_position_emb: 是否需要空间位置嵌入
            need_temporal_weight: 是否需要时间权重
            self_attn_mask: 自注意力掩码类型
            image_scale: 图像缩放因子
            processor: 注意力处理器
            remove_femb_non_linear: 是否移除帧嵌入的非线性激活
        """
        super().__init__()
        logger.info("初始化TransformerTemporalModel")

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.in_channels = in_channels

        # 创建GroupNorm层用于输入归一化
        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        logger.debug(f"创建GroupNorm层，组数: {norm_num_groups}, 通道数: {in_channels}")

        # 输入投影层，将输入通道映射到内部维度
        self.proj_in = nn.Linear(in_channels, inner_dim)
        logger.debug(f"创建输入投影层，输入维度: {in_channels}, 输出维度: {inner_dim}")

        # 2. 定义时间位置嵌入
        self.frame_emb_proj = torch.nn.Linear(femb_channels, inner_dim)
        self.remove_femb_non_linear = remove_femb_non_linear
        if not remove_femb_non_linear:
            self.nonlinearity = nn.SiLU()
        logger.debug(f"创建帧嵌入投影层，输入维度: {femb_channels}, 输出维度: {inner_dim}")

        # spatial_position_emb 使用femb_的参数配置
        self.need_spatial_position_emb = need_spatial_position_emb
        if need_spatial_position_emb:
            self.spatial_position_emb_proj = torch.nn.Linear(femb_channels, inner_dim)
            logger.debug("创建空间位置嵌入投影层")
            
        # 3. 定义transformer块
        # TODO： 该实现方式不好，待优化
        # TODO: bad implementation, need to be optimized
        self.need_ipadapter = False
        self.cross_attn_temporal_cond = False
        self.allow_xformers = allow_xformers
        if processor is not None and isinstance(processor, BaseIPAttnProcessor):
            self.cross_attn_temporal_cond = True
            self.allow_xformers = False
            if "NonParam" not in processor.__class__.__name__:
                self.need_ipadapter = True
            logger.debug("检测到IPAdapter处理器，配置相关参数")

        # 创建Transformer块列表
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    double_self_attention=double_self_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                    allow_xformers=allow_xformers,
                    only_cross_attention=only_cross_attention,
                    cross_attn_temporal_cond=self.need_ipadapter,
                    image_scale=image_scale,
                    processor=processor,
                )
                for d in range(num_layers)
            ]
        )
        logger.info(f"创建了 {num_layers} 个Transformer块")

        # 输出投影层，将内部维度映射回输入通道数
        self.proj_out = nn.Linear(inner_dim, in_channels)
        logger.debug(f"创建输出投影层，输入维度: {inner_dim}, 输出维度: {in_channels}")

        # 时间权重参数，用于控制时间信息的融合程度
        self.need_temporal_weight = need_temporal_weight
        if need_temporal_weight:
            self.temporal_weight = nn.Parameter(
                torch.tensor(
                    [
                        1e-5,
                    ]
                )
            )  # initialize parameter with 0
            logger.debug("创建时间权重参数")
            
        self.skip_temporal_layers = False  # 是否跳过时间层
        self.keep_content_condition = keep_content_condition
        self.self_attn_mask = self_attn_mask
        self.only_cross_attention = only_cross_attention
        self.double_self_attention = double_self_attention
        self.cross_attention_dim = cross_attention_dim
        self.image_scale = image_scale
        
        # 将最后一层参数初始化为0，使卷积块成为恒等映射
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        logger.debug("初始化输出投影层权重为0")

    def forward(
        self,
        hidden_states,
        femb,
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        num_frames=1,
        cross_attention_kwargs=None,
        sample_index: torch.LongTensor = None,
        vision_conditon_frames_sample_index: torch.LongTensor = None,
        spatial_position_emb: torch.Tensor = None,
        return_dict: bool = True,
    ):
        """
        前向传播函数

        Args:
            hidden_states (当为离散时，为形状 `(batch size, num latent pixels)` 的 `torch.LongTensor`。
                当为连续时，为形状 `(batch size, channel, height, width)` 的 `torch.FloatTensor`): 输入
                hidden_states
            encoder_hidden_states (形状为 `(batch size, encoder_hidden_states dim)` 的 `torch.LongTensor`，*可选*):
                用于交叉注意力层的条件嵌入。如果未提供，则交叉注意力默认为
                自注意力。
            timestep ( `torch.long`，*可选*):
                作为AdaLayerNorm嵌入应用的可选时间步。用于指示去噪步骤。
            class_labels (形状为 `(batch size, num classes)` 的 `torch.LongTensor`，*可选*):
                作为AdaLayerZeroNorm嵌入应用的可选类标签。用于指示类标签
                条件。
            return_dict (`bool`，*可选*，默认为 `True`):
                是否返回 [`models.unet_2d_condition.UNet2DConditionOutput`] 而不是普通元组。

        Returns:
            [`~models.transformer_2d.TransformerTemporalModelOutput`] 或 `tuple`:
            如果 `return_dict` 为 True，则为 [`~models.transformer_2d.TransformerTemporalModelOutput`]，否则为 `tuple`。
            返回元组时，第一个元素是样本张量。
        """
        logger.debug(f"开始前向传播，输入hidden_states形状: {hidden_states.shape}")
        
        # 如果跳过时间层，直接返回输入
        if self.skip_temporal_layers is True:
            logger.info("跳过时间层，直接返回输入")
            if not return_dict:
                return (hidden_states,)

            return TransformerTemporalModelOutput(sample=hidden_states)

        # 1. 输入处理
        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames
        logger.debug(f"输入形状分析 - batch_frames: {batch_frames}, channel: {channel}, height: {height}, width: {width}, batch_size: {batch_size}")

        # 重新排列张量维度: (b*t, c, h, w) -> (b, c, t, h, w)
        hidden_states = rearrange(
            hidden_states, "(b t) c h w -> b c t h w", b=batch_size
        )
        logger.debug(f"重新排列hidden_states维度后形状: {hidden_states.shape}")
        
        residual = hidden_states  # 保存残差连接用的原始输入

        # 对输入进行归一化
        hidden_states = self.norm(hidden_states)
        logger.debug("完成输入归一化")

        # 重新排列张量维度: (b, c, t, h, w) -> (b*h*w, t, c)
        hidden_states = rearrange(hidden_states, "b c t h w -> (b h w) t c")
        logger.debug(f"为Transformer准备的hidden_states形状: {hidden_states.shape}")

        # 输入投影
        hidden_states = self.proj_in(hidden_states)
        logger.debug(f"输入投影后形状: {hidden_states.shape}")

        # 2 位置嵌入处理
        # 改编自 https://github.com/huggingface/diffusers/blob/v0.16.1/src/diffusers/models/resnet.py#L574
        if not self.remove_femb_non_linear:
            femb = self.nonlinearity(femb)
            logger.debug("应用帧嵌入非线性激活")
            
        # 投影帧嵌入到内部维度
        femb = self.frame_emb_proj(femb)
        logger.debug(f"帧嵌入投影后形状: {femb.shape}")
        
        # 对齐并重复张量以匹配hidden_states的批次大小
        femb = align_repeat_tensor_single_dim(femb, hidden_states.shape[0], dim=0)
        logger.debug(f"对齐后的帧嵌入形状: {femb.shape}")
        
        # 将帧嵌入添加到hidden_states
        hidden_states = hidden_states + femb
        logger.debug("完成帧嵌入与hidden_states的融合")

        # 3. Transformer块处理
        if (
            (self.only_cross_attention or not self.double_self_attention)
            and self.cross_attention_dim is not None
            and encoder_hidden_states is not None
        ):
            # 对encoder_hidden_states进行对齐和重复处理
            encoder_hidden_states = align_repeat_tensor_single_dim(
                encoder_hidden_states,
                hidden_states.shape[0],
                dim=0,
                n_src_base_length=batch_size,
            )
            logger.debug(f"处理encoder_hidden_states，处理后形状: {encoder_hidden_states.shape}")

        # 依次通过每个Transformer块
        for i, block in enumerate(self.transformer_blocks):
            logger.debug(f"处理第 {i} 个Transformer块")
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )
            logger.debug(f"第 {i} 个Transformer块输出形状: {hidden_states.shape}")

        # 4. 输出处理
        # 输出投影
        hidden_states = self.proj_out(hidden_states)
        logger.debug(f"输出投影后形状: {hidden_states.shape}")
        
        # 重新排列张量维度: (b*h*w, t, c) -> (b, c, t, h, w)
        hidden_states = rearrange(
            hidden_states, "(b h w) t c -> b c t h w", b=batch_size, h=height, w=width
        ).contiguous()
        logger.debug(f"重新排列输出维度后形状: {hidden_states.shape}")

        # 保留condition对应的frames，便于保持前序内容帧，提升一致性
        # keep the frames corresponding to the condition to maintain the previous content frames and improve consistency
        if (
            vision_conditon_frames_sample_index is not None
            and self.keep_content_condition
        ):
            logger.debug("应用内容条件保持机制")
            mask = torch.ones_like(hidden_states, device=hidden_states.device)
            mask = batch_index_fill(
                mask, dim=2, index=vision_conditon_frames_sample_index, value=0
            )
            logger.debug(f"创建掩码，形状: {mask.shape}")
            
            if self.need_temporal_weight:
                output = (
                    residual + torch.abs(self.temporal_weight) * mask * hidden_states
                )
                logger.debug("应用时间权重和掩码")
            else:
                output = residual + mask * hidden_states
                logger.debug("应用掩码")
        else:
            if self.need_temporal_weight:
                output = residual + torch.abs(self.temporal_weight) * hidden_states
                logger.debug("应用时间权重")
            else:
                output = residual + mask * hidden_states
                logger.debug("应用残差连接")

        # 最终重新排列: (b, c, t, h, w) -> (b*t, c, h, w)
        output = rearrange(output, "b c t h w -> (b t) c h w")
        logger.debug(f"最终输出形状: {output.shape}")
        
        if not return_dict:
            logger.info("前向传播完成，返回元组格式结果")
            return (output,)

        logger.info("前向传播完成，返回TransformerTemporalModelOutput格式结果")
        return TransformerTemporalModelOutput(sample=output)