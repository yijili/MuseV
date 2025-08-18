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

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
from diffusers.models.attention_processor import Attention, AttnProcessor
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F
import xformers
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.models.unet_2d_condition import (
    UNet2DConditionModel,
    UNet2DConditionOutput,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.constants import USE_PEFT_BACKEND
from diffusers.utils.deprecation_utils import deprecate
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_utils import ModelMixin, load_state_dict
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    BaseOutput,
    deprecate,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    PositionNet,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin


from ..data.data_util import align_repeat_tensor_single_dim
from .unet_3d_condition import UNet3DConditionModel
from .attention import BasicTransformerBlock, IPAttention
from .unet_2d_blocks import (
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    get_down_block,
    get_up_block,
)

from . import Model_Register


logger = logging.getLogger(__name__)


@Model_Register.register
class ReferenceNet2D(UNet2DConditionModel, nn.Module):
    """继承 UNet2DConditionModel. 新增功能，类似controlnet 返回模型中间特征，用于后续作用
        Inherit Unet2DConditionModel. Add new functions, similar to controlnet, return the intermediate features of the model for subsequent effects
    Args:
        UNet2DConditionModel (_type_): _description_
    """

    _supports_gradient_checkpointing = True
    print_idx = 0

    @register_to_config
    def __init__(
        self,
        sample_size: int | None = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: str | None = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        only_cross_attention: bool | Tuple[bool] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int | Tuple[int] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0,
        act_fn: str = "silu",
        norm_num_groups: int | None = 32,
        norm_eps: float = 0.00001,
        cross_attention_dim: int | Tuple[int] = 1280,
        transformer_layers_per_block: int | Tuple[int] | Tuple[Tuple] = 1,
        reverse_transformer_layers_per_block: Tuple[Tuple[int]] | None = None,
        encoder_hid_dim: int | None = None,
        encoder_hid_dim_type: str | None = None,
        attention_head_dim: int | Tuple[int] = 8,
        num_attention_heads: int | Tuple[int] | None = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: str | None = None,
        addition_embed_type: str | None = None,
        addition_time_embed_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1,
        time_embedding_type: str = "positional",
        time_embedding_dim: int | None = None,
        time_embedding_act_fn: str | None = None,
        timestep_post_act: str | None = None,
        time_cond_proj_dim: int | None = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: int | None = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: bool | None = None,
        cross_attention_norm: str | None = None,
        addition_embed_type_num_heads=64,
        need_self_attn_block_embs: bool = False,
        need_block_embs: bool = False,
    ):
        """
        初始化 ReferenceNet2D 模型
        
        Args:
            sample_size: 输入样本大小
            in_channels: 输入通道数
            out_channels: 输出通道数
            center_input_sample: 是否居中输入样本
            flip_sin_to_cos: 是否翻转正弦到余弦
            freq_shift: 频率偏移
            down_block_types: 下采样块类型列表
            mid_block_type: 中间块类型
            up_block_types: 上采样块类型列表
            only_cross_attention: 是否仅使用交叉注意力
            block_out_channels: 每个块的输出通道数
            layers_per_block: 每个块的层数
            downsample_padding: 下采样填充
            mid_block_scale_factor: 中间块缩放因子
            dropout: Dropout 比例
            act_fn: 激活函数类型
            norm_num_groups: 归一化组数
            norm_eps: 归一化 epsilon 值
            cross_attention_dim: 交叉注意力维度
            transformer_layers_per_block: 每个块中的 transformer 层数
            reverse_transformer_layers_per_block: 反向 transformer 层数
            encoder_hid_dim: 编码器隐藏维度
            encoder_hid_dim_type: 编码器隐藏维度类型
            attention_head_dim: 注意力头维度
            num_attention_heads: 注意力头数量
            dual_cross_attention: 是否使用双重交叉注意力
            use_linear_projection: 是否使用线性投影
            class_embed_type: 类别嵌入类型
            addition_embed_type: 额外嵌入类型
            addition_time_embed_dim: 额外时间嵌入维度
            num_class_embeds: 类别嵌入数量
            upcast_attention: 是否上转换注意力
            resnet_time_scale_shift: ResNet 时间缩放偏移
            resnet_skip_time_act: ResNet 是否跳过时间激活
            resnet_out_scale_factor: ResNet 输出缩放因子
            time_embedding_type: 时间嵌入类型
            time_embedding_dim: 时间嵌入维度
            time_embedding_act_fn: 时间嵌入激活函数
            timestep_post_act: 时间步后处理激活函数
            time_cond_proj_dim: 时间条件投影维度
            conv_in_kernel: 输入卷积核大小
            conv_out_kernel: 输出卷积核大小
            projection_class_embeddings_input_dim: 投影类别嵌入输入维度
            attention_type: 注意力类型
            class_embeddings_concat: 是否连接类别嵌入
            mid_block_only_cross_attention: 中间块是否仅使用交叉注意力
            cross_attention_norm: 交叉注意力归一化类型
            addition_embed_type_num_heads: 额外嵌入类型头数
            need_self_attn_block_embs: 是否需要自注意力块嵌入
            need_block_embs: 是否需要块嵌入
        """
        super().__init__()
        logger.info("初始化 ReferenceNet2D 模型")

        self.sample_size = sample_size

        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )

        # 如果未定义 num_attention_heads（大多数模型的情况），则默认为 attention_head_dim
        num_attention_heads = num_attention_heads or attention_head_dim

        # 检查输入参数
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"必须提供相同数量的 `down_block_types` 和 `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"必须提供相同数量的 [block_out_channels] 和 `down_block_types`. block_out_channels: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(
            only_cross_attention
        ) != len(down_block_types):
            raise ValueError(
                f"必须提供相同数量的 `only_cross_attention` 和 `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(
            down_block_types
        ):
            raise ValueError(
                f"必须提供相同数量的 num_attention_heads 和 `down_block_types`. num_attention_heads: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(
            down_block_types
        ):
            raise ValueError(
                f"必须提供相同数量的 attention_head_dim 和 `down_block_types`. attention_head_dim: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(
            down_block_types
        ):
            raise ValueError(
                f"必须提供相同数量的 cross_attention_dim 和 `down_block_types`. cross_attention_dim: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(
            down_block_types
        ):
            raise ValueError(
                f"必须提供相同数量的 layers_per_block 和 `down_block_types`. layers_per_block: {layers_per_block}. `down_block_types`: {down_block_types}."
            )
        if (
            isinstance(transformer_layers_per_block, list)
            and reverse_transformer_layers_per_block is None
        ):
            for layer_number_per_block in transformer_layers_per_block:
                if isinstance(layer_number_per_block, list):
                    raise ValueError(
                        "如果使用非对称 UNet，必须提供 'reverse_transformer_layers_per_block`。"
                    )

        # 输入层
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=conv_in_kernel,
            padding=conv_in_padding,
        )
        logger.debug(f"创建输入卷积层: in_channels={in_channels}, out_channels={block_out_channels[0]}")

        # 时间嵌入
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(
                    f"`time_embed_dim` 应该能被 2 整除, 但是是 {time_embed_dim}."
                )
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos, freq_shift
            )
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} 不存在。请确保使用 `fourier` 或 `positional` 之一。"
            )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )
        logger.debug(f"创建时间嵌入层: timestep_input_dim={timestep_input_dim}, time_embed_dim={time_embed_dim}")

        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info(
                "encoder_hid_dim_type 默认为 'text_proj' 因为定义了 `encoder_hid_dim`。"
            )

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"当 `encoder_hid_dim_type` 设置为 {encoder_hid_dim_type} 时，必须定义 `encoder_hid_dim`。"
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            # image_embed_dim 不一定必须是 cross_attention_dim。为了不使 __init__ 太混乱
            # 它们在这里设置为 cross_attention_dim，因为这是目前唯一使用的情况
            # 当 `addition_embed_type == "text_image_proj"` 时 (Kadinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2
            self.encoder_hid_proj = ImageProjection(
                image_embed_dim=encoder_hid_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"encoder_hid_dim_type: {encoder_hid_dim_type} 必须是 None, 'text_proj' 或 'text_image_proj'。"
            )
        else:
            self.encoder_hid_proj = None

        # 类别嵌入
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(
                timestep_input_dim, time_embed_dim, act_fn=act_fn
            )
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' 需要设置 `projection_class_embeddings_input_dim`"
                )
            # 投影 `class_embed_type` 与时间步 `class_embed_type` 相同，除了
            # 1. `class_labels` 输入不会首先转换为正弦嵌入
            # 2. 它从任意输入维度进行投影。
            self.class_embedding = TimestepEmbedding(
                projection_class_embeddings_input_dim, time_embed_dim
            )
        elif class_embed_type == "simple_projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'simple_projection' 需要设置 `projection_class_embeddings_input_dim`"
                )
            self.class_embedding = nn.Linear(
                projection_class_embeddings_input_dim, time_embed_dim
            )
        else:
            self.class_embedding = None

        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim,
                time_embed_dim,
                num_heads=addition_embed_type_num_heads,
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim 和 image_embed_dim 不一定必须是 cross_attention_dim。为了不使 __init__ 太混乱
            # 它们在这里设置为 cross_attention_dim，因为这是目前唯一使用的情况
            # 当 `addition_embed_type == "text_image"` 时 (Kadinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim,
                image_embed_dim=cross_attention_dim,
                time_embed_dim=time_embed_dim,
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(
                addition_time_embed_dim, flip_sin_to_cos, freq_shift
            )
            self.add_embedding = TimestepEmbedding(
                projection_class_embeddings_input_dim, time_embed_dim
            )
        elif addition_embed_type == "image":
            # Kandinsky 2.2
            self.add_embedding = ImageTimeEmbedding(
                image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "image_hint":
            # Kandinsky 2.2 ControlNet
            self.add_embedding = ImageHintTimeEmbedding(
                image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type is not None:
            raise ValueError(
                f"addition_embed_type: {addition_embed_type} 必须是 None, 'text' 或 'text_image'。"
            )

        if time_embedding_act_fn is None:
            self.time_embed_act = None
        else:
            self.time_embed_act = get_activation(time_embedding_act_fn)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention

            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(
                down_block_types
            )

        if class_embeddings_concat:
            # 时间嵌入与类别嵌入连接。传递给下、中、上块的时间嵌入维度是常规时间嵌入维度的两倍
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # 下采样块
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i]
                if attention_head_dim[i] is not None
                else output_channel,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)
            logger.debug(f"创建下采样块 {i}: type={down_block_type}")

        # 中间块
        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                transformer_layers_per_block=transformer_layers_per_block[-1],
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                attention_type=attention_type,
            )
        elif mid_block_type == "UNetMidBlock2DSimpleCrossAttn":
            self.mid_block = UNetMidBlock2DSimpleCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=cross_attention_dim[-1],
                attention_head_dim=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                skip_time_act=resnet_skip_time_act,
                only_cross_attention=mid_block_only_cross_attention,
                cross_attention_norm=cross_attention_norm,
            )
        elif mid_block_type == "UNetMidBlock2D":
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                dropout=dropout,
                num_layers=0,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                add_attention=False,
            )
        elif mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"未知的 mid_block_type : {mid_block_type}")
        
        if self.mid_block is not None:
            logger.debug(f"创建中间块: type={mid_block_type}")

        # 计算上采样层数
        self.num_upsamplers = 0

        # 上采样块
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            # 为除最后一层外的所有层添加上采样块
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resolution_idx=i,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i]
                if attention_head_dim[i] is not None
                else output_channel,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
            logger.debug(f"创建上采样块 {i}: type={up_block_type}")

        # 输出层
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0],
                num_groups=norm_num_groups,
                eps=norm_eps,
            )

            self.conv_act = get_activation(act_fn)

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=conv_out_kernel,
            padding=conv_out_padding,
        )

        if attention_type in ["gated", "gated-text-image"]:
            positive_len = 768
            if isinstance(cross_attention_dim, int):
                positive_len = cross_attention_dim
            elif isinstance(cross_attention_dim, tuple) or isinstance(
                cross_attention_dim, list
            ):
                positive_len = cross_attention_dim[0]

            feature_type = "text-only" if attention_type == "gated" else "text-image"
            self.position_net = PositionNet(
                positive_len=positive_len,
                out_dim=cross_attention_dim,
                feature_type=feature_type,
            )
        self.need_block_embs = need_block_embs
        self.need_self_attn_block_embs = need_self_attn_block_embs

        # 仅使用 referencenet 的某些层，其他层设为 None
        self.conv_norm_out = None
        self.conv_act = None
        self.conv_out = None

        self.up_blocks[-1].attentions[-1].proj_out = None
        self.up_blocks[-1].attentions[-1].transformer_blocks[-1].attn1 = None
        self.up_blocks[-1].attentions[-1].transformer_blocks[-1].attn2 = None
        self.up_blocks[-1].attentions[-1].transformer_blocks[-1].norm2 = None
        self.up_blocks[-1].attentions[-1].transformer_blocks[-1].ff = None
        self.up_blocks[-1].attentions[-1].transformer_blocks[-1].norm3 = None
        if not self.need_self_attn_block_embs:
            self.up_blocks = None

        self.insert_spatial_self_attn_idx()
        logger.info("ReferenceNet2D 模型初始化完成")

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        # update new paramestes start
        num_frames: int = None,
        return_ndim: int = 5,
        # update new paramestes end
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        ReferenceNet2D 模型的前向传播方法。

        Args:
            sample (`torch.FloatTensor`):
                带噪声的输入张量，形状为 `(batch, channel, height, width)`。
            timestep (`torch.FloatTensor` 或 `float` 或 `int`): 去噪的时间步数。
            encoder_hidden_states (`torch.FloatTensor`):
                编码器隐藏状态，形状为 `(batch, sequence_length, feature_dim)`。
            class_labels (`torch.Tensor`, *可选*, 默认为 `None`):
                可选的类别标签用于条件控制。它们的嵌入将与时间步嵌入相加。
            timestep_cond: (`torch.Tensor`, *可选*, 默认为 `None`):
                时间步的条件嵌入。如果提供，嵌入将与通过 `self.time_embedding` 层传递的样本相加以获得时间步嵌入。
            attention_mask (`torch.Tensor`, *可选*, 默认为 `None`):
                应用于 `encoder_hidden_states` 的注意力掩码，形状为 `(batch, key_tokens)`。如果为 `1` 则保留掩码，
                否则如果为 `0` 则丢弃。掩码将被转换为偏置，为"丢弃"标记对应的注意力分数添加大的负值。
            cross_attention_kwargs (`dict`, *可选*):
                如果指定，则传递给 `AttentionProcessor` 的 kwargs 字典，定义在
                diffusers.models.attention_processor https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py 中的 `self.processor` 下。
            added_cond_kwargs: (`dict`, *可选*):
                包含额外嵌入的 kwargs 字典，如果指定则添加到传递给 UNet 块的嵌入中。
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *可选*):
                如果指定，则添加到下 unet 块残差的张量元组。
            mid_block_additional_residual: (`torch.Tensor`, *可选*):
                如果指定，则添加到中间 unet 块输出的张量。
            encoder_attention_mask (`torch.Tensor`):
                应用于 `encoder_hidden_states` 的交叉注意力掩码，形状为 `(batch, sequence_length)`。
                如果为 `True` 则保留掩码，否则如果为 `False` 则丢弃。掩码将被转换为偏置，
                为"丢弃"标记对应的注意力分数添加大的负值。
            return_dict (`bool`, *可选*, 默认为 `True`):
                是否返回 [`~models.unet_2d_condition.UNet2DConditionOutput`] 而不是普通元组。
            cross_attention_kwargs (`dict`, *可选*):
                如果指定，则传递给 [`AttnProcessor`] 的 kwargs 字典。
            added_cond_kwargs: (`dict`, *可选*):
                包含额外嵌入的 kwargs 字典，如果指定则添加到传递给 UNet 块的嵌入中。
            down_block_additional_residuals (`tuple` of `torch.Tensor`, *可选*):
                从下块到上块的 UNet 长跳跃连接中要添加的额外残差，例如来自 ControlNet 侧模型的残差
            mid_block_additional_residual (`torch.Tensor`, *可选*):
                要添加到 UNet 中间块输出的额外残差，例如来自 ControlNet 侧模型的残差
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *可选*):
                要添加到 UNet 下块内部的额外残差，例如来自 T2I-Adapter 侧模型的残差
            num_frames: 帧数
            return_ndim: 返回张量的维度数

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] 或 `tuple`:
                如果 `return_dict` 为 True，返回 [`~models.unet_2d_condition.UNet2DConditionOutput`]，否则
                返回一个元组，其中第一个元素是样本张量。
        """

        logger.debug(f"开始前向传播: sample.shape={sample.shape}, timestep={timestep}")
        
        # 默认情况下，样本至少应该是整体上采样因子的倍数。
        # 整体上采样因子等于 2 ** (# 上采样层数)。
        # 但是，如有必要，上采样插值输出大小可以在运行时强制适应任何上采样大小。
        default_overall_up_factor = 2**self.num_upsamplers

        # 当样本不是 `default_overall_up_factor` 的倍数时，应转发上采样大小
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # 转发上采样大小以强制插值输出大小。
                forward_upsample_size = True
                break

        # 确保 attention_mask 是偏置，并给它一个单独的 query_tokens 维度
        # 期望的掩码形状:
        #   [batch, key_tokens]
        # 添加单独的 query_tokens 维度:
        #   [batch,                    1, key_tokens]
        # 这有助于将其作为偏置广播到注意力分数，注意力分数将采用以下形状之一:
        #   [batch,  heads, query_tokens, key_tokens] (例如 torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (例如 xformers 或经典 attn)
        if attention_mask is not None:
            # 假设掩码表示为:
            #   (1 = 保留,      0 = 丢弃)
            # 将掩码转换为可以添加到注意力分数的偏置:
            #       (保留 = +0,     丢弃 = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 以相同方式将 encoder_attention_mask 转换为偏置
        if encoder_attention_mask is not None:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(sample.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. 如有必要，居中输入
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. 时间处理
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: 这需要 CPU 和 GPU 之间的同步。所以尽量传递张量形式的时间步
            # 这将是 `match` 语句的一个很好的用例 (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # 以与 ONNX/Core ML 兼容的方式广播到批次维度
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        logger.debug(f"时间投影完成: t_emb.shape={t_emb.shape}")

        # `Timesteps` 不包含任何权重，将始终返回 f32 张量
        # 但 time_embedding 实际上可能在 fp16 中运行。所以我们需要在这里进行转换。
        # 可能有更好的封装方法。
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        logger.debug(f"时间嵌入完成: emb.shape={emb.shape}")
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "当 num_class_embeds > 0 时，应提供 class_labels"
                )

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` 不包含任何权重，将始终返回 f32 张量
                # 可能有更好的封装方法。
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - 风格
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} 的配置参数 `addition_embed_type` 设置为 'text_image'，这要求在 `added_cond_kwargs` 中传递关键字参数 `image_embeds`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            # SDXL - 风格
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} 的配置参数 `addition_embed_type` 设置为 'text_time'，这要求在 `added_cond_kwargs` 中传递关键字参数 `text_embeds`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} 的配置参数 `addition_embed_type` 设置为 'text_time'，这要求在 `added_cond_kwargs` 中传递关键字参数 `time_ids`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - 风格
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} 的配置参数 `addition_embed_type` 设置为 'image'，这要求在 `added_cond_kwargs` 中传递关键字参数 `image_embeds`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - 风格
            if (
                "image_embeds" not in added_cond_kwargs
                or "hint" not in added_cond_kwargs
            ):
                raise ValueError(
                    f"{self.__class__} 的配置参数 `addition_embed_type` 设置为 'image_hint'，这要求在 `added_cond_kwargs` 中传递关键字参数 `image_embeds` 和 `hint`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if (
            self.encoder_hid_proj is not None
            and self.config.encoder_hid_dim_type == "text_proj"
        ):
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif (
            self.encoder_hid_proj is not None
            and self.config.encoder_hid_dim_type == "text_image_proj"
        ):
            # Kadinsky 2.1 - 风格
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} 的配置参数 `encoder_hid_dim_type` 设置为 'text_image_proj'，这要求在 `added_conditions` 中传递关键字参数 `image_embeds`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(
                encoder_hidden_states, image_embeds
            )
        elif (
            self.encoder_hid_proj is not None
            and self.config.encoder_hid_dim_type == "image_proj"
        ):
            # Kandinsky 2.2 - 风格
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} 的配置参数 `encoder_hid_dim_type` 设置为 'image_proj'，这要求在 `added_conditions` 中传递关键字参数 `image_embeds`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        elif (
            self.encoder_hid_proj is not None
            and self.config.encoder_hid_dim_type == "ip_image_proj"
        ):
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} 的配置参数 `encoder_hid_dim_type` 设置为 'ip_image_proj'，这要求在 `added_conditions` 中传递关键字参数 `image_embeds`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            image_embeds = self.encoder_hid_proj(image_embeds).to(
                encoder_hidden_states.dtype
            )
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, image_embeds], dim=1
            )

        # need_self_attn_block_embs
        # 初始化
        # 或在unet中运算中会不断 append self_attn_blocks_embs，用完需要清理，
        if self.need_self_attn_block_embs:
            self_attn_block_embs = [None] * self.self_attn_num
        else:
            self_attn_block_embs = None
        # 2. 预处理
        sample = self.conv_in(sample)
        if self.print_idx == 0:
            logger.debug(f"经过 conv_in 后的 sample 均值={sample.mean()}")
        # 2.5 GLIGEN 位置网络
        if (
            cross_attention_kwargs is not None
            and cross_attention_kwargs.get("gligen", None) is not None
        ):
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {
                "objs": self.position_net(**gligen_args)
            }

        # 3. 下采样
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )
        if USE_PEFT_BACKEND:
            # 通过为每个 PEFT 层设置 `lora_scale` 来加权 lora 层
            scale_lora_layers(self, lora_scale)

        is_controlnet = (
            mid_block_additional_residual is not None
            and down_block_additional_residuals is not None
        )
        # 使用新参数 down_intrablock_additional_residuals 用于 T2I-Adapters，以区别于 controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # 为向后兼容性保留旧用法，其中
        #       T2I-Adapter 和 ControlNet 都使用 down_block_additional_residuals 参数
        #       但只能使用其中一个
        if (
            not is_adapter
            and mid_block_additional_residual is None
            and down_block_additional_residuals is not None
        ):
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "传递块内残差连接与 `down_block_additional_residuals` 已弃用 \
                       并将在 diffusers 1.3.0 中移除。`down_block_additional_residuals` 应该仅用于 \
                       ControlNet。请确保改用 `down_intrablock_additional_residuals`。 ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for i_downsample_block, downsample_block in enumerate(self.down_blocks):
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                # 对于 t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals[
                        "additional_residuals"
                    ] = down_intrablock_additional_residuals.pop(0)
                if self.print_idx == 0:
                    logger.debug(
                        f"下采样块 {i_downsample_block} sample 均值={sample.mean()}"
                    )
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                    self_attn_block_embs=self_attn_block_embs,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    scale=lora_scale,
                    self_attn_block_embs=self_attn_block_embs,
                )
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = (
                    down_block_res_sample + down_block_additional_residual
                )
                new_down_block_res_samples = new_down_block_res_samples + (
                    down_block_res_sample,
                )

            down_block_res_samples = new_down_block_res_samples

        # 更新代码开始
        def reshape_return_emb(tmp_emb):
            if return_ndim == 4:
                return tmp_emb
            elif return_ndim == 5:
                return rearrange(tmp_emb, "(b t) c h w-> b c t h w", t=num_frames)
            else:
                raise ValueError(
                    f"reshape_emb 仅支持 4, 5 但给定 {return_ndim}"
                )

        if self.need_block_embs:
            return_down_block_res_samples = [
                reshape_return_emb(tmp_emb) for tmp_emb in down_block_res_samples
            ]
        else:
            return_down_block_res_samples = None
        # 更新代码结束

        # 4. 中间块
        if self.mid_block is not None:
            if (
                hasattr(self.mid_block, "has_cross_attention")
                and self.mid_block.has_cross_attention
            ):
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    self_attn_block_embs=self_attn_block_embs,
                )
            else:
                sample = self.mid_block(sample, emb)

            # 支持 T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        if self.need_block_embs:
            return_mid_block_res_samples = reshape_return_emb(sample)
            logger.debug(
                f"return_mid_block_res_samples, is_leaf={return_mid_block_res_samples.is_leaf}, requires_grad={return_mid_block_res_samples.requires_grad}"
            )
        else:
            return_mid_block_res_samples = None

        if self.up_blocks is not None:
            # 更新代码结束

            # 5. 上采样
            for i, upsample_block in enumerate(self.up_blocks):
                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[
                    : -len(upsample_block.resnets)
                ]

                # 如果我们尚未到达最终块且需要转发
                # 上采样大小，我们在这里进行
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if (
                    hasattr(upsample_block, "has_cross_attention")
                    and upsample_block.has_cross_attention
                ):
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        self_attn_block_embs=self_attn_block_embs,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                        scale=lora_scale,
                        self_attn_block_embs=self_attn_block_embs,
                    )

        # 更新代码开始
        if self.need_block_embs or self.need_self_attn_block_embs:
            if self_attn_block_embs is not None:
                self_attn_block_embs = [
                    reshape_return_emb(tmp_emb=tmp_emb)
                    for tmp_emb in self_attn_block_embs
                ]
            self.print_idx += 1
            logger.debug("返回块嵌入和自注意力嵌入")
            return (
                return_down_block_res_samples,
                return_mid_block_res_samples,
                self_attn_block_embs,
            )

        if not self.need_block_embs and not self.need_self_attn_block_embs:
            # 6. 后处理
            if self.conv_norm_out:
                sample = self.conv_norm_out(sample)
                sample = self.conv_act(sample)
            sample = self.conv_out(sample)

            if USE_PEFT_BACKEND:
                # 从每个 PEFT 层中移除 `lora_scale`
                unscale_lora_layers(self, lora_scale)
            self.print_idx += 1
            if not return_dict:
                logger.debug("返回样本元组")
                return (sample,)

            logger.debug("返回 UNet2DConditionOutput")
            return UNet2DConditionOutput(sample=sample)

    def insert_spatial_self_attn_idx(self):
        """
        插入空间自注意力索引
        """
        attns, basic_transformers = self.spatial_self_attns
        self.self_attn_num = len(attns)
        for i, (name, layer) in enumerate(attns):
            logger.debug(f"{self.__class__.__name__}, {i}, {name}, {type(layer)}")
            if layer is not None:
                layer.spatial_self_attn_idx = i
        for i, (name, layer) in enumerate(basic_transformers):
            logger.debug(f"{self.__class__.__name__}, {i}, {name}, {type(layer)}")
            if layer is not None:
                layer.spatial_self_attn_idx = i

    @property
    def spatial_self_attns(
        self,
    ) -> List[Tuple[str, Attention]]:
        """
        获取空间自注意力层
        
        Returns:
            空间自注意力层列表
        """
        attns, spatial_transformers = self.get_self_attns(
            include="attentions", exclude="temp_attentions"
        )
        attns = sorted(attns)
        spatial_transformers = sorted(spatial_transformers)
        return attns, spatial_transformers

    def get_self_attns(
        self, include: str = None, exclude: str = None
    ) -> List[Tuple[str, Attention]]:
        r"""
        返回:
            注意力层字典: 包含模型中使用的所有注意力层的字典，按键名索引。
        """
        # 递归设置
        attns = []
        spatial_transformers = []

        def fn_recursive_add_attns(
            name: str,
            module: torch.nn.Module,
            attns: List[Tuple[str, Attention]],
            spatial_transformers: List[Tuple[str, BasicTransformerBlock]],
        ):
            is_target = False
            if isinstance(module, BasicTransformerBlock) and hasattr(module, "attn1"):
                is_target = True
                if include is not None:
                    is_target = include in name
                if exclude is not None:
                    is_target = exclude not in name
            if is_target:
                attns.append([f"{name}.attn1", module.attn1])
                spatial_transformers.append([f"{name}", module])
            for sub_name, child in module.named_children():
                fn_recursive_add_attns(
                    f"{name}.{sub_name}", child, attns, spatial_transformers
                )

            return attns

        for name, module in self.named_children():
            fn_recursive_add_attns(name, module, attns, spatial_transformers)

        return attns, spatial_transformers


class ReferenceNet3D(UNet3DConditionModel):
    """继承 UNet3DConditionModel， 用于提取中间emb用于后续作用。
        Inherit Unet3DConditionModel, used to extract the middle emb for subsequent actions.
    Args:
        UNet3DConditionModel (_type_): _description_
    """

    pass