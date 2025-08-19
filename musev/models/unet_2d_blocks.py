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
from typing import Any, Dict, Literal, Optional, Tuple, Union, List
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import is_torch_version
from diffusers.utils.torch_utils import apply_freeu
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import (
    Attention,
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
)
from diffusers.models.dual_transformer_2d import DualTransformer2DModel
from diffusers.models.normalization import AdaGroupNorm
from diffusers.models.resnet import (
    Downsample2D,
    FirDownsample2D,
    FirUpsample2D,
    KDownsample2D,
    KUpsample2D,
    ResnetBlock2D,
    Upsample2D,
)
from diffusers.models.unet_2d_blocks import (
    AttnDownBlock2D,
    AttnDownEncoderBlock2D,
    AttnSkipDownBlock2D,
    AttnSkipUpBlock2D,
    AttnUpBlock2D,
    AttnUpDecoderBlock2D,
    DownEncoderBlock2D,
    KCrossAttnDownBlock2D,
    KCrossAttnUpBlock2D,
    KDownBlock2D,
    KUpBlock2D,
    ResnetDownsampleBlock2D,
    ResnetUpsampleBlock2D,
    SimpleCrossAttnDownBlock2D,
    SimpleCrossAttnUpBlock2D,
    SkipDownBlock2D,
    SkipUpBlock2D,
    UpDecoderBlock2D,
)

from .transformer_2d import Transformer2DModel

# 获取日志记录器实例
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.basicConfig(level=logging.info)

def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    downsample_type: Optional[str] = None,
    dropout: float = 0.0,
):
    """
    根据指定的下采样块类型创建并返回相应的下采样块
    
    Args:
        down_block_type (str): 下采样块的类型
        num_layers (int): 块中层的数量
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        temb_channels (int): 时间嵌入通道数
        add_downsample (bool): 是否添加下采样层
        resnet_eps (float): ResNet中的epsilon值
        resnet_act_fn (str): ResNet激活函数类型
        transformer_layers_per_block (int): 每个Transformer块的层数
        num_attention_heads (Optional[int]): 注意力头数量
        resnet_groups (Optional[int]): ResNet组归一化中的组数
        cross_attention_dim (Optional[int]): 交叉注意力维度
        downsample_padding (Optional[int]): 下采样填充大小
        dual_cross_attention (bool): 是否使用双重交叉注意力
        use_linear_projection (bool): 是否使用线性投影
        only_cross_attention (bool): 是否仅使用交叉注意力
        upcast_attention (bool): 是否上转换注意力
        resnet_time_scale_shift (str): ResNet时间缩放偏移类型
        attention_type (str): 注意力类型
        resnet_skip_time_act (bool): ResNet是否跳过时间激活
        resnet_out_scale_factor (float): ResNet输出缩放因子
        cross_attention_norm (Optional[str]): 交叉注意力归一化类型
        attention_head_dim (Optional[int]): 注意力头维度
        downsample_type (Optional[str]): 下采样类型
        dropout (float): Dropout概率

    Returns:
        nn.Module: 对应类型的下采样块实例
    """
    # 如果未定义注意力头维度，则默认使用注意力头数量
    if attention_head_dim is None:
        logger.warn(
            f"建议在调用 get_down_block 时提供 attention_head_dim 参数。默认将 attention_head_dim 设置为 {num_attention_heads}。"
        )
        attention_head_dim = num_attention_heads

    # 处理块类型前缀
    down_block_type = (
        down_block_type[7:]
        if down_block_type.startswith("UNetRes")
        else down_block_type
    )
    
    logger.debug(f"创建下采样块类型: {down_block_type}")
    
    # 根据块类型创建相应的下采样块
    if down_block_type == "DownBlock2D":
        logger.debug("初始化 DownBlock2D 下采样块")
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "ResnetDownsampleBlock2D":
        logger.debug("初始化 ResnetDownsampleBlock2D 下采样块")
        return ResnetDownsampleBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            output_scale_factor=resnet_out_scale_factor,
        )
    elif down_block_type == "AttnDownBlock2D":
        logger.debug("初始化 AttnDownBlock2D 下采样块")
        if add_downsample is False:
            downsample_type = None
        else:
            downsample_type = downsample_type or "conv"  # 默认为 'conv'
        return AttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            downsample_type=downsample_type,
        )
    elif down_block_type == "CrossAttnDownBlock2D":
        logger.debug("初始化 CrossAttnDownBlock2D 下采样块")
        if cross_attention_dim is None:
            raise ValueError(
                "CrossAttnDownBlock2D 必须指定 cross_attention_dim 参数"
            )
        return CrossAttnDownBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
        )
    elif down_block_type == "SimpleCrossAttnDownBlock2D":
        logger.debug("初始化 SimpleCrossAttnDownBlock2D 下采样块")
        if cross_attention_dim is None:
            raise ValueError(
                "SimpleCrossAttnDownBlock2D 必须指定 cross_attention_dim 参数"
            )
        return SimpleCrossAttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            output_scale_factor=resnet_out_scale_factor,
            only_cross_attention=only_cross_attention,
            cross_attention_norm=cross_attention_norm,
        )
    elif down_block_type == "SkipDownBlock2D":
        logger.debug("初始化 SkipDownBlock2D 下采样块")
        return SkipDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "AttnSkipDownBlock2D":
        logger.debug("初始化 AttnSkipDownBlock2D 下采样块")
        return AttnSkipDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "DownEncoderBlock2D":
        logger.debug("初始化 DownEncoderBlock2D 下采样块")
        return DownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "AttnDownEncoderBlock2D":
        logger.debug("初始化 AttnDownEncoderBlock2D 下采样块")
        return AttnDownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "KDownBlock2D":
        logger.debug("初始化 KDownBlock2D 下采样块")
        return KDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
        )
    elif down_block_type == "KCrossAttnDownBlock2D":
        logger.debug("初始化 KCrossAttnDownBlock2D 下采样块")
        return KCrossAttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            add_self_attention=True if not add_downsample else False,
        )
    
    # 如果没有匹配的块类型，则抛出异常
    raise ValueError(f"下采样块类型 {down_block_type} 不存在。")


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resolution_idx: Optional[int] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    upsample_type: Optional[str] = None,
    dropout: float = 0.0,
) -> nn.Module:
    """
    根据指定的上采样块类型创建并返回相应的上采样块
    
    Args:
        up_block_type (str): 上采样块的类型
        num_layers (int): 块中层的数量
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        prev_output_channel (int): 前一个块的输出通道数
        temb_channels (int): 时间嵌入通道数
        add_upsample (bool): 是否添加上采样层
        resnet_eps (float): ResNet中的epsilon值
        resnet_act_fn (str): ResNet激活函数类型
        resolution_idx (Optional[int]): 分辨率索引
        transformer_layers_per_block (int): 每个Transformer块的层数
        num_attention_heads (Optional[int]): 注意力头数量
        resnet_groups (Optional[int]): ResNet组归一化中的组数
        cross_attention_dim (Optional[int]): 交叉注意力维度
        dual_cross_attention (bool): 是否使用双重交叉注意力
        use_linear_projection (bool): 是否使用线性投影
        only_cross_attention (bool): 是否仅使用交叉注意力
        upcast_attention (bool): 是否上转换注意力
        resnet_time_scale_shift (str): ResNet时间缩放偏移类型
        attention_type (str): 注意力类型
        resnet_skip_time_act (bool): ResNet是否跳过时间激活
        resnet_out_scale_factor (float): ResNet输出缩放因子
        cross_attention_norm (Optional[str]): 交叉注意力归一化类型
        attention_head_dim (Optional[int]): 注意力头维度
        upsample_type (Optional[str]): 上采样类型
        dropout (float): Dropout概率

    Returns:
        nn.Module: 对应类型的上采样块实例
    """
    # 如果未定义注意力头维度，则默认使用注意力头数量
    if attention_head_dim is None:
        logger.warn(
            f"建议在调用 get_up_block时提供 attention_head_dim 参数。默认将 attention_head_dim 设置为 {num_attention_heads}。"
        )
        attention_head_dim = num_attention_heads

    # 处理块类型前缀
    up_block_type = (
        up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    )
    
    logger.debug(f"创建上采样块类型: {up_block_type}")
    
    # 根据块类型创建相应的上采样块
    if up_block_type == "UpBlock2D":
        logger.debug("初始化 UpBlock2D 上采样块")
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "ResnetUpsampleBlock2D":
        logger.debug("初始化 ResnetUpsampleBlock2D 上采样块")
        return ResnetUpsampleBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            output_scale_factor=resnet_out_scale_factor,
        )
    elif up_block_type == "CrossAttnUpBlock2D":
        logger.debug("初始化 CrossAttnUpBlock2D 上采样块")
        if cross_attention_dim is None:
            raise ValueError(
                "CrossAttnUpBlock2D 必须指定 cross_attention_dim 参数"
            )
        return CrossAttnUpBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
        )
    elif up_block_type == "SimpleCrossAttnUpBlock2D":
        logger.debug("初始化 SimpleCrossAttnUpBlock2D 上采样块")
        if cross_attention_dim is None:
            raise ValueError(
                "SimpleCrossAttnUpBlock2D 必须指定 cross_attention_dim 参数"
            )
        return SimpleCrossAttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            output_scale_factor=resnet_out_scale_factor,
            only_cross_attention=only_cross_attention,
            cross_attention_norm=cross_attention_norm,
        )
    elif up_block_type == "AttnUpBlock2D":
        logger.debug("初始化 AttnUpBlock2D 上采样块")
        if add_upsample is False:
            upsample_type = None
        else:
            upsample_type = upsample_type or "conv"  # 默认为 'conv'

        return AttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            upsample_type=upsample_type,
        )
    elif up_block_type == "SkipUpBlock2D":
        logger.debug("初始化 SkipUpBlock2D 上采样块")
        return SkipUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "AttnSkipUpBlock2D":
        logger.debug("初始化 AttnSkipUpBlock2D 上采样块")
        return AttnSkipUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "UpDecoderBlock2D":
        logger.debug("初始化 UpDecoderBlock2D 上采样块")
        return UpDecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
        )
    elif up_block_type == "AttnUpDecoderBlock2D":
        logger.debug("初始化 AttnUpDecoderBlock2D 上采样块")
        return AttnUpDecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
        )
    elif up_block_type == "KUpBlock2D":
        logger.debug("初始化 KUpBlock2D 上采样块")
        return KUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
        )
    elif up_block_type == "KCrossAttnUpBlock2D":
        logger.debug("初始化 KCrossAttnUpBlock2D 上采样块")
        return KCrossAttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
        )

    # 如果没有匹配的块类型，则抛出异常
    raise ValueError(f"上采样块类型 {up_block_type} 不存在。")


class UNetMidBlock2D(nn.Module):
    """
    一个2D UNet中间块 UNetMidBlock2D，包含多个残差块和可选的注意力块。

    Args:
        in_channels (`int`): 输入通道数。
        temb_channels (`int`): 时间嵌入通道数。
        dropout (`float`, *optional*, 默认为 0.0): Dropout率。
        num_layers (`int`, *optional*, 默认为 1): 残差块的数量。
        resnet_eps (`float`, *optional*, 1e-6 ): ResNet块的epsilon值。
        resnet_time_scale_shift (`str`, *optional*, 默认为 `default`):
            应用于时间嵌入的归一化类型。这有助于提高模型在具有长期时间依赖性的任务上的性能。
        resnet_act_fn (`str`, *optional*, 默认为 `swish`): ResNet块的激活函数。
        resnet_groups (`int`, *optional*, 默认为 32):
            ResNet块中组归一化层使用的组数。
        attn_groups (`Optional[int]`, *optional*, 默认为 None): 注意力块的组数。
        resnet_pre_norm (`bool`, *optional*, 默认为 `True`):
            是否在ResNet块中使用预归一化。
        add_attention (`bool`, *optional*, 默认为 `True`): 是否添加注意力块。
        attention_head_dim (`int`, *optional*, 默认为 1):
            单个注意力头的维度。注意力头的数量基于此值和输入通道数确定。
        output_scale_factor (`float`, *optional*, 默认为 1.0): 输出缩放因子。

    Returns:
        `torch.FloatTensor`: 最后一个残差块的输出，是一个形状为 `(batch_size,
        in_channels, height, width)` 的张量。

    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        """
        初始化UNet中间块
        
        Args:
            in_channels (int): 输入通道数
            temb_channels (int): 时间嵌入通道数
            dropout (float): Dropout概率
            num_layers (int): 残差块层数
            resnet_eps (float): ResNet epsilon值
            resnet_time_scale_shift (str): 时间缩放偏移类型
            resnet_act_fn (str): ResNet激活函数
            resnet_groups (int): ResNet组数
            attn_groups (Optional[int]): 注意力组数
            resnet_pre_norm (bool): 是否预归一化
            add_attention (bool): 是否添加注意力
            attention_head_dim (int): 注意力头维度
            output_scale_factor (float): 输出缩放因子
        """
        super().__init__()
        logger.debug(f"初始化 UNetMidBlock2D，输入通道数: {in_channels}, 时间嵌入通道数: {temb_channels}")
        
        # 设置组归一化组数
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.add_attention = add_attention

        # 设置注意力组数
        if attn_groups is None:
            attn_groups = (
                resnet_groups if resnet_time_scale_shift == "default" else None
            )

        # 至少有一个resnet块
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        # 如果未指定注意力头维度，则默认使用输入通道数
        if attention_head_dim is None:
            logger.warn(
                f"不建议传递 `attention_head_dim=None`。默认将 attention_head_dim 设置为 `in_channels`: {in_channels}。"
            )
            attention_head_dim = in_channels

        # 创建注意力块和残差块
        for _ in range(num_layers):
            if self.add_attention:
                logger.debug("添加注意力块到 UNetMidBlock2D")
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=temb_channels
                        if resnet_time_scale_shift == "spatial"
                        else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        self_attn_block_embs: Optional[List[torch.Tensor]] = None,
        self_attn_block_embs_mode: Literal["read", "write"] = "write",
    ) -> torch.FloatTensor:
        """
        前向传播函数
        
        Args:
            hidden_states (torch.FloatTensor): 隐藏状态输入
            temb (Optional[torch.FloatTensor]): 时间嵌入
            self_attn_block_embs (Optional[List[torch.Tensor]]): 自注意力块嵌入
            self_attn_block_embs_mode (Literal["read", "write"]): 自注意力块嵌入模式

        Returns:
            torch.FloatTensor: 前向传播后的输出张量
        """
        logger.debug(f"UNetMidBlock2D 前向传播开始，输入形状: {hidden_states.shape}")
        
        # 第一个resnet块
        hidden_states = self.resnets[0](hidden_states, temb)
        logger.debug(f"经过第一个 ResNet 块后形状: {hidden_states.shape}")
        
        # 遍历注意力块和resnet块
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                logger.debug("处理注意力块")
                hidden_states = attn(
                    hidden_states,
                    temb=temb,
                    self_attn_block_embs=self_attn_block_embs,
                    self_attn_block_embs_mode=self_attn_block_embs_mode,
                )
                logger.debug(f"经过注意力块后形状: {hidden_states.shape}")
            logger.debug("处理 ResNet 块")
            hidden_states = resnet(hidden_states, temb)
            logger.debug(f"经过 ResNet 块后形状: {hidden_states.shape}")

        logger.debug(f"UNetMidBlock2D 前向传播完成，输出形状: {hidden_states.shape}")
        return hidden_states


class UNetMidBlock2DCrossAttn(nn.Module):
    """
    UNet中间块，包含交叉注意力机制
    """
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        """
        初始化带有交叉注意力的UNet中间块
        
        Args:
            in_channels (int): 输入通道数
            temb_channels (int): 时间嵌入通道数
            dropout (float): Dropout概率
            num_layers (int): 层数
            transformer_layers_per_block (Union[int, Tuple[int]]): 每个Transformer块的层数
            resnet_eps (float): ResNet epsilon值
            resnet_time_scale_shift (str): 时间缩放偏移类型
            resnet_act_fn (str): ResNet激活函数
            resnet_groups (int): ResNet组数
            resnet_pre_norm (bool): 是否预归一化
            num_attention_heads (int): 注意力头数
            output_scale_factor (float): 输出缩放因子
            cross_attention_dim (int): 交叉注意力维度
            dual_cross_attention (bool): 是否使用双重交叉注意力
            use_linear_projection (bool): 是否使用线性投影
            upcast_attention (bool): 是否上转换注意力
            attention_type (str): 注意力类型
        """
        super().__init__()
        logger.debug(f"初始化 UNetMidBlock2DCrossAttn，输入通道数: {in_channels}")

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )

        # 支持每个块的可变Transformer层数
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        # 至少有一个resnet块
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        # 创建注意力块和resnet块
        for i in range(num_layers):
            if not dual_cross_attention:
                logger.debug(f"添加 Transformer2DModel 注意力块 {i}")
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                logger.debug(f"添加 DualTransformer2DModel 注意力块 {i}")
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        self_attn_block_embs: Optional[List[torch.Tensor]] = None,
        self_attn_block_embs_mode: Literal["read", "write"] = "write",
    ) -> torch.FloatTensor:
        """
        前向传播函数
        
        Args:
            hidden_states (torch.FloatTensor): 隐藏状态输入
            temb (Optional[torch.FloatTensor]): 时间嵌入
            encoder_hidden_states (Optional[torch.FloatTensor]): 编码器隐藏状态
            attention_mask (Optional[torch.FloatTensor]): 注意力掩码
            cross_attention_kwargs (Optional[Dict[str, Any]]): 交叉注意力参数
            encoder_attention_mask (Optional[torch.FloatTensor]): 编码器注意力掩码
            self_attn_block_embs (Optional[List[torch.Tensor]]): 自注意力块嵌入
            self_attn_block_embs_mode (Literal["read", "write"]): 自注意力块嵌入模式

        Returns:
            torch.FloatTensor: 前向传播后的输出张量
        """
        logger.debug(f"UNetMidBlock2DCrossAttn 前向传播开始，输入形状: {hidden_states.shape}")
        
        # 获取LoRA缩放因子
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )
        
        # 第一个resnet块
        hidden_states = self.resnets[0](hidden_states, temb, scale=lora_scale)
        logger.debug(f"经过第一个 ResNet 块后形状: {hidden_states.shape}")
        
        # 遍历注意力块和resnet块
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if self.training and self.gradient_checkpointing:
                logger.debug("使用梯度检查点技术")

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                    self_attn_block_embs=self_attn_block_embs,
                    self_attn_block_embs_mode=self_attn_block_embs_mode,
                )[0]
                logger.debug(f"经过注意力块后形状: {hidden_states.shape}")
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                logger.debug(f"经过 ResNet 块后形状: {hidden_states.shape}")
            else:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                    self_attn_block_embs=self_attn_block_embs,
                )[0]
                logger.debug(f"经过注意力块后形状: {hidden_states.shape}")
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                logger.debug(f"经过 ResNet 块后形状: {hidden_states.shape}")

        logger.debug(f"UNetMidBlock2DCrossAttn 前向传播完成，输出形状: {hidden_states.shape}")
        return hidden_states


class UNetMidBlock2DSimpleCrossAttn(nn.Module):
    """
    简单交叉注意力UNet中间块
    """
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        skip_time_act: bool = False,
        only_cross_attention: bool = False,
        cross_attention_norm: Optional[str] = None,
    ):
        """
        初始化简单交叉注意力UNet中间块
        
        Args:
            in_channels (int): 输入通道数
            temb_channels (int): 时间嵌入通道数
            dropout (float): Dropout概率
            num_layers (int): 层数
            resnet_eps (float): ResNet epsilon值
            resnet_time_scale_shift (str): 时间缩放偏移类型
            resnet_act_fn (str): ResNet激活函数
            resnet_groups (int): ResNet组数
            resnet_pre_norm (bool): 是否预归一化
            attention_head_dim (int): 注意力头维度
            output_scale_factor (float): 输出缩放因子
            cross_attention_dim (int): 交叉注意力维度
            skip_time_act (bool): 是否跳过时间激活
            only_cross_attention (bool): 是否仅使用交叉注意力
            cross_attention_norm (Optional[str]): 交叉注意力归一化类型
        """
        super().__init__()
        logger.debug(f"初始化 UNetMidBlock2DSimpleCrossAttn，输入通道数: {in_channels}")

        self.has_cross_attention = True

        self.attention_head_dim = attention_head_dim
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )

        self.num_heads = in_channels // self.attention_head_dim

        # 至少有一个resnet块
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                skip_time_act=skip_time_act,
            )
        ]
        attentions = []

        # 创建注意力块和resnet块
        for _ in range(num_layers):
            processor = (
                AttnAddedKVProcessor2_0()
                if hasattr(F, "scaled_dot_product_attention")
                else AttnAddedKVProcessor()
            )

            attentions.append(
                Attention(
                    query_dim=in_channels,
                    cross_attention_dim=in_channels,
                    heads=self.num_heads,
                    dim_head=self.attention_head_dim,
                    added_kv_proj_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    bias=True,
                    upcast_softmax=True,
                    only_cross_attention=only_cross_attention,
                    cross_attention_norm=cross_attention_norm,
                    processor=processor,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    skip_time_act=skip_time_act,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        self_attn_block_embs: Optional[List[torch.Tensor]] = None,
        self_attn_block_embs_mode: Literal["read", "write"] = "write",
    ) -> torch.FloatTensor:
        """
        前向传播函数
        
        Args:
            hidden_states (torch.FloatTensor): 隐藏状态输入
            temb (Optional[torch.FloatTensor]): 时间嵌入
            encoder_hidden_states (Optional[torch.FloatTensor]): 编码器隐藏状态
            attention_mask (Optional[torch.FloatTensor]): 注意力掩码
            cross_attention_kwargs (Optional[Dict[str, Any]]): 交叉注意力参数
            encoder_attention_mask (Optional[torch.FloatTensor]): 编码器注意力掩码
            self_attn_block_embs (Optional[List[torch.Tensor]]): 自注意力块嵌入
            self_attn_block_embs_mode (Literal["read", "write"]): 自注意力块嵌入模式

        Returns:
            torch.FloatTensor: 前向传播后的输出张量
        """
        logger.debug(f"UNetMidBlock2DSimpleCrossAttn 前向传播开始，输入形状: {hidden_states.shape}")
        
        # 处理交叉注意力参数
        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )
        lora_scale = cross_attention_kwargs.get("scale", 1.0)

        # 设置注意力掩码
        if attention_mask is None:
            # 如果定义了 encoder_hidden_states: 我们正在进行交叉注意力，所以应该使用交叉注意力掩码
            mask = None if encoder_hidden_states is None else encoder_attention_mask
        else:
            # 当定义了 attention_mask: 我们甚至不检查 encoder_attention_mask
            # 这是为了与 UnCLIP 兼容，UnCLIP 使用 'attention_mask' 参数进行交叉注意力掩码
            mask = attention_mask

        # 第一个resnet块
        hidden_states = self.resnets[0](hidden_states, temb, scale=lora_scale)
        logger.debug(f"经过第一个 ResNet 块后形状: {hidden_states.shape}")
        
        # 遍历注意力块和resnet块
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            # 注意力块处理
            logger.debug("处理注意力块")
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=mask,
                **cross_attention_kwargs,
                self_attn_block_embs=self_attn_block_embs,
                self_attn_block_embs_mode=self_attn_block_embs_mode,
            )
            logger.debug(f"经过注意力块后形状: {hidden_states.shape}")

            # resnet块处理
            logger.debug("处理 ResNet 块")
            hidden_states = resnet(hidden_states, temb, scale=lora_scale)
            logger.debug(f"经过 ResNet 块后形状: {hidden_states.shape}")

        logger.debug(f"UNetMidBlock2DSimpleCrossAttn 前向传播完成，输出形状: {hidden_states.shape}")
        return hidden_states


class CrossAttnDownBlock2D(nn.Module):
    """
    带有交叉注意力的下采样块
    """
    print_idx = 0

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        """
        初始化带有交叉注意力的下采样块
        
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            temb_channels (int): 时间嵌入通道数
            dropout (float): Dropout概率
            num_layers (int): 层数
            transformer_layers_per_block (Union[int, Tuple[int]]): 每个Transformer块的层数
            resnet_eps (float): ResNet epsilon值
            resnet_time_scale_shift (str): 时间缩放偏移类型
            resnet_act_fn (str): ResNet激活函数
            resnet_groups (int): ResNet组数
            resnet_pre_norm (bool): 是否预归一化
            num_attention_heads (int): 注意力头数
            cross_attention_dim (int): 交叉注意力维度
            output_scale_factor (float): 输出缩放因子
            downsample_padding (int): 下采样填充大小
            add_downsample (bool): 是否添加下采样
            dual_cross_attention (bool): 是否使用双重交叉注意力
            use_linear_projection (bool): 是否使用线性投影
            only_cross_attention (bool): 是否仅使用交叉注意力
            upcast_attention (bool): 是否上转换注意力
            attention_type (str): 注意力类型
        """
        super().__init__()
        logger.debug(f"初始化 CrossAttnDownBlock2D，输入通道数: {in_channels}，输出通道数: {out_channels}")
        
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        # 支持每个块的可变Transformer层数
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        # 创建resnet块和注意力块
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            logger.debug(f"添加 ResNet 块 {i}")
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                logger.debug(f"添加 Transformer2DModel 注意力块 {i}")
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                logger.debug(f"添加 DualTransformer2DModel 注意力块 {i}")
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        # 添加下采样层
        if add_downsample:
            logger.debug("添加下采样层")
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        additional_residuals: Optional[torch.FloatTensor] = None,
        self_attn_block_embs: Optional[List[torch.Tensor]] = None,
        self_attn_block_embs_mode: Literal["read", "write"] = "write",
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        """
        前向传播函数
        
        Args:
            hidden_states (torch.FloatTensor): 隐藏状态输入
            temb (Optional[torch.FloatTensor]): 时间嵌入
            encoder_hidden_states (Optional[torch.FloatTensor]): 编码器隐藏状态
            attention_mask (Optional[torch.FloatTensor]): 注意力掩码
            cross_attention_kwargs (Optional[Dict[str, Any]]): 交叉注意力参数
            encoder_attention_mask (Optional[torch.FloatTensor]): 编码器注意力掩码
            additional_residuals (Optional[torch.FloatTensor]): 额外残差
            self_attn_block_embs (Optional[List[torch.Tensor]]): 自注意力块嵌入
            self_attn_block_embs_mode (Literal["read", "write"]): 自注意力块嵌入模式

        Returns:
            Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]: 输出张量和中间状态元组
        """
        logger.debug(f"CrossAttnDownBlock2D 前向传播开始，输入形状: {hidden_states.shape}")
        
        output_states = ()

        # 获取LoRA缩放因子
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )

        blocks = list(zip(self.resnets, self.attentions))

        # 遍历resnet块和注意力块
        for i, (resnet, attn) in enumerate(blocks):
            if self.training and self.gradient_checkpointing:
                logger.debug(f"使用梯度检查点技术处理块 {i}")

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                if self.print_idx == 0:
                    logger.debug(f"unet3d after resnet {hidden_states.mean()}")

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                    self_attn_block_embs=self_attn_block_embs,
                    self_attn_block_embs_mode=self_attn_block_embs_mode,
                )[0]
            else:
                logger.debug(f"处理 ResNet 块 {i}")
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                if self.print_idx == 0:
                    logger.debug(f"unet3d after resnet {hidden_states.mean()}")
                logger.debug(f"处理注意力块 {i}")
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                    self_attn_block_embs=self_attn_block_embs,
                    self_attn_block_embs_mode=self_attn_block_embs_mode,
                )[0]

            # 将额外残差应用于最后一对resnet和注意力块的输出
            if i == len(blocks) - 1 and additional_residuals is not None:
                logger.debug("应用额外残差")
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)

        # 处理下采样
        if self.downsamplers is not None:
            logger.debug("处理下采样")
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale=lora_scale)

            output_states = output_states + (hidden_states,)

        self.print_idx += 1
        logger.debug(f"CrossAttnDownBlock2D 前向传播完成，输出形状: {hidden_states.shape}")
        return hidden_states, output_states


class DownBlock2D(nn.Module):
    """
    2D下采样块
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        """
        初始化2D下采样块
        
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            temb_channels (int): 时间嵌入通道数
            dropout (float): Dropout概率
            num_layers (int): 层数
            resnet_eps (float): ResNet epsilon值
            resnet_time_scale_shift (str): 时间缩放偏移类型
            resnet_act_fn (str): ResNet激活函数
            resnet_groups (int): ResNet组数
            resnet_pre_norm (bool): 是否预归一化
            output_scale_factor (float): 输出缩放因子
            add_downsample (bool): 是否添加下采样
            downsample_padding (int): 下采样填充大小
        """
        super().__init__()
        logger.debug(f"初始化 DownBlock2D，输入通道数: {in_channels}，输出通道数: {out_channels}")
        
        resnets = []

        # 创建resnet块
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            logger.debug(f"添加 ResNet 块 {i}")
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        # 添加下采样层
        if add_downsample:
            logger.debug("添加下采样层")
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        self_attn_block_embs: Optional[List[torch.Tensor]] = None,
        self_attn_block_embs_mode: Literal["read", "write"] = "write",
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        """
        前向传播函数
        
        Args:
            hidden_states (torch.FloatTensor): 隐藏状态输入
            temb (Optional[torch.FloatTensor]): 时间嵌入
            scale (float): 缩放因子
            self_attn_block_embs (Optional[List[torch.Tensor]]): 自注意力块嵌入
            self_attn_block_embs_mode (Literal["read", "write"]): 自注意力块嵌入模式

        Returns:
            Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]: 输出张量和中间状态元组
        """
        logger.debug(f"DownBlock2D 前向传播开始，输入形状: {hidden_states.shape}")
        
        output_states = ()

        # 遍历resnet块
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:
                logger.debug("使用梯度检查点技术")

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                logger.debug("处理 ResNet 块")
                hidden_states = resnet(hidden_states, temb, scale=scale)

            output_states = output_states + (hidden_states,)

        # 处理下采样
        if self.downsamplers is not None:
            logger.debug("处理下采样")
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale=scale)

            output_states = output_states + (hidden_states,)

        logger.debug(f"DownBlock2D 前向传播完成，输出形状: {hidden_states.shape}")
        return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
    """
    带有交叉注意力的上采样块
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        """
        初始化带有交叉注意力的上采样块
        
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            prev_output_channel (int): 前一个块的输出通道数
            temb_channels (int): 时间嵌入通道数
            resolution_idx (Optional[int]): 分辨率索引
            dropout (float): Dropout概率
            num_layers (int): 层数
            transformer_layers_per_block (Union[int, Tuple[int]]): 每个Transformer块的层数
            resnet_eps (float): ResNet epsilon值
            resnet_time_scale_shift (str): 时间缩放偏移类型
            resnet_act_fn (str): ResNet激活函数
            resnet_groups (int): ResNet组数
            resnet_pre_norm (bool): 是否预归一化
            num_attention_heads (int): 注意力头数
            cross_attention_dim (int): 交叉注意力维度
            output_scale_factor (float): 输出缩放因子
            add_upsample (bool): 是否添加上采样
            dual_cross_attention (bool): 是否使用双重交叉注意力
            use_linear_projection (bool): 是否使用线性投影
            only_cross_attention (bool): 是否仅使用交叉注意力
            upcast_attention (bool): 是否上转换注意力
            attention_type (str): 注意力类型
        """
        super().__init__()
        logger.debug(f"初始化 CrossAttnUpBlock2D，输入通道数: {in_channels}，输出通道数: {out_channels}")
        
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        # 支持每个块的可变Transformer层数
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        # 创建resnet块和注意力块
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            logger.debug(f"添加 ResNet 块 {i}")
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                logger.debug(f"添加 Transformer2DModel 注意力块 {i}")
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                logger.debug(f"添加 DualTransformer2DModel 注意力块 {i}")
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        # 添加上采样层
        if add_upsample:
            logger.debug("添加上采样层")
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        self_attn_block_embs: Optional[List[torch.Tensor]] = None,
        self_attn_block_embs_mode: Literal["read", "write"] = "write",
    ) -> torch.FloatTensor:
        """
        前向传播函数
        
        Args:
            hidden_states (torch.FloatTensor): 隐藏状态输入
            res_hidden_states_tuple (Tuple[torch.FloatTensor, ...]): 残差隐藏状态元组
            temb (Optional[torch.FloatTensor]): 时间嵌入
            encoder_hidden_states (Optional[torch.FloatTensor]): 编码器隐藏状态
            cross_attention_kwargs (Optional[Dict[str, Any]]): 交叉注意力参数
            upsample_size (Optional[int]): 上采样大小
            attention_mask (Optional[torch.FloatTensor]): 注意力掩码
            encoder_attention_mask (Optional[torch.FloatTensor]): 编码器注意力掩码
            self_attn_block_embs (Optional[List[torch.Tensor]]): 自注意力块嵌入
            self_attn_block_embs_mode (Literal["read", "write"]): 自注意力块嵌入模式

        Returns:
            torch.FloatTensor: 输出张量
        """
        logger.debug(f"CrossAttnUpBlock2D 前向传播开始，输入形状: {hidden_states.shape}")
        
        # 获取LoRA缩放因子
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        # 遍历resnet块和注意力块
        for resnet, attn in zip(self.resnets, self.attentions):
            # 弹出残差隐藏状态
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: 仅在前两个阶段操作
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:
                logger.debug("使用梯度检查点技术")

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                logger.debug(f"经过 ResNet 块后形状: {hidden_states.shape}")
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                    self_attn_block_embs=self_attn_block_embs,
                    self_attn_block_embs_mode=self_attn_block_embs_mode,
                )[0]
                logger.debug(f"经过注意力块后形状: {hidden_states.shape}")
            else:
                logger.debug("处理 ResNet 块")
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                logger.debug(f"经过 ResNet 块后形状: {hidden_states.shape}")
                logger.debug("处理注意力块")
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                    self_attn_block_embs=self_attn_block_embs,
                )[0]
                logger.debug(f"经过注意力块后形状: {hidden_states.shape}")

        # 处理上采样
        if self.upsamplers is not None:
            logger.debug("处理上采样")
            for upsampler in self.upsamplers:
                hidden_states = upsampler(
                    hidden_states, upsample_size, scale=lora_scale
                )

        logger.debug(f"CrossAttnUpBlock2D 前向传播完成，输出形状: {hidden_states.shape}")
        return hidden_states


class UpBlock2D(nn.Module):
    """
    2D上采样块
    """
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        """
        初始化2D上采样块
        
        Args:
            in_channels (int): 输入通道数
            prev_output_channel (int): 前一个块的输出通道数
            out_channels (int): 输出通道数
            temb_channels (int): 时间嵌入通道数
            resolution_idx (Optional[int]): 分辨率索引
            dropout (float): Dropout概率
            num_layers (int): 层数
            resnet_eps (float): ResNet epsilon值
            resnet_time_scale_shift (str): 时间缩放偏移类型
            resnet_act_fn (str): ResNet激活函数
            resnet_groups (int): ResNet组数
            resnet_pre_norm (bool): 是否预归一化
            output_scale_factor (float): 输出缩放因子
            add_upsample (bool): 是否添加上采样
        """
        super().__init__()
        logger.debug(f"初始化 UpBlock2D，输入通道数: {in_channels}，输出通道数: {out_channels}")
        
        resnets = []

        # 创建resnet块
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            logger.debug(f"添加 ResNet 块 {i}")
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        # 添加上采样层
        if add_upsample:
            logger.debug("添加上采样层")
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        upsample_size: Optional[int] = None,
        scale: float = 1.0,
        self_attn_block_embs: Optional[List[torch.Tensor]] = None,
        self_attn_block_embs_mode: Literal["read", "write"] = "write",
    ) -> torch.FloatTensor:
        """
        前向传播函数
        
        Args:
            hidden_states (torch.FloatTensor): 隐藏状态输入
            res_hidden_states_tuple (Tuple[torch.FloatTensor, ...]): 残差隐藏状态元组
            temb (Optional[torch.FloatTensor]): 时间嵌入
            upsample_size (Optional[int]): 上采样大小
            scale (float): 缩放因子
            self_attn_block_embs (Optional[List[torch.Tensor]]): 自注意力块嵌入
            self_attn_block_embs_mode (Literal["read", "write"]): 自注意力块嵌入模式

        Returns:
            torch.FloatTensor: 输出张量
        """
        logger.debug(f"UpBlock2D 前向传播开始，输入形状: {hidden_states.shape}")
        
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        # 遍历resnet块
        for resnet in self.resnets:
            # 弹出残差隐藏状态
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: 仅在前两个阶段操作
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:
                logger.debug("使用梯度检查点技术")

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                logger.debug("处理 ResNet 块")
                hidden_states = resnet(hidden_states, temb, scale=scale)
                logger.debug(f"经过 ResNet 块后形状: {hidden_states.shape}")

        # 处理上采样
        if self.upsamplers is not None:
            logger.debug("处理上采样")
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size, scale=scale)
            logger.debug(f"经过上采样后形状: {hidden_states.shape}")

        logger.debug(f"UpBlock2D 前向传播完成，输出形状: {hidden_states.shape}")
        return hidden_states