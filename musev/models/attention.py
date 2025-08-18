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

# Adapted from https://github.com/huggingface/diffusers/blob/64bf5d33b7ef1b1deac256bed7bd99b55020c4e0/src/diffusers/models/attention.py
from __future__ import annotations
from copy import deepcopy

from typing import Any, Dict, List, Literal, Optional, Callable, Tuple
import logging
from einops import rearrange

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention as DiffusersAttention
from diffusers.models.attention import (
    BasicTransformerBlock as DiffusersBasicTransformerBlock,
    AdaLayerNormZero,
    AdaLayerNorm,
    FeedForward,
)
from diffusers.models.attention_processor import AttnProcessor

from .attention_processor import IPAttention, BaseIPAttnProcessor


logger = logging.getLogger(__name__)


def not_use_xformers_anyway(
    use_memory_efficient_attention_xformers: bool,
    attention_op: Optional[Callable] = None,
):
    """
    禁用xformers注意力机制的函数
    
    Args:
        use_memory_efficient_attention_xformers: 是否使用内存高效的注意力机制
        attention_op: 注意力操作函数
    
    Returns:
        None
    """
    logger.debug("禁用xformers注意力机制")
    return None


@maybe_allow_in_graph
class BasicTransformerBlock(DiffusersBasicTransformerBlock):
    """
    基础Transformer块，扩展了Diffusers的BasicTransformerBlock类，支持IP-Adapter等功能
    """
    print_idx = 0

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int | None = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        attention_type: str = "default",
        allow_xformers: bool = True,
        cross_attn_temporal_cond: bool = False,
        image_scale: float = 1.0,
        processor: AttnProcessor | None = None,
        ip_adapter_cross_attn: bool = False,
        need_t2i_facein: bool = False,
        need_t2i_ip_adapter_face: bool = False,
    ):
        """
        初始化BasicTransformerBlock
        
        Args:
            dim: 隐藏层维度
            num_attention_heads: 注意力头数
            attention_head_dim: 每个注意力头的维度
            dropout: Dropout比率
            cross_attention_dim: 交叉注意力维度
            activation_fn: 激活函数类型
            num_embeds_ada_norm: AdaLayerNorm的嵌入数
            attention_bias: 是否使用注意力偏置
            only_cross_attention: 是否仅使用交叉注意力
            double_self_attention: 是否使用双重自注意力
            upcast_attention: 是否提升注意力计算精度
            norm_elementwise_affine: 是否在归一化中使用元素级仿射变换
            norm_type: 归一化类型
            final_dropout: 是否在最后使用dropout
            attention_type: 注意力类型
            allow_xformers: 是否允许使用xformers
            cross_attn_temporal_cond: 是否使用交叉注意力时间条件
            image_scale: 图像缩放因子
            processor: 注意力处理器
            ip_adapter_cross_attn: 是否使用IP-Adapter交叉注意力
            need_t2i_facein: 是否需要T2I FaceIn功能
            need_t2i_ip_adapter_face: 是否需要T2I IP-Adapter面部功能
        """
        logger.debug(f"初始化BasicTransformerBlock: dim={dim}, num_attention_heads={num_attention_heads}, attention_head_dim={attention_head_dim}")
        
        if not only_cross_attention and double_self_attention:
            cross_attention_dim = None
        super().__init__(
            dim,
            num_attention_heads,
            attention_head_dim,
            dropout,
            cross_attention_dim,
            activation_fn,
            num_embeds_ada_norm,
            attention_bias,
            only_cross_attention,
            double_self_attention,
            upcast_attention,
            norm_elementwise_affine,
            norm_type,
            final_dropout,
            attention_type,
        )

        # 初始化第一个注意力层（自注意力）
        self.attn1 = IPAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            cross_attn_temporal_cond=cross_attn_temporal_cond,
            image_scale=image_scale,
            ip_adapter_dim=cross_attention_dim
            if only_cross_attention
            else attention_head_dim,
            facein_dim=cross_attention_dim
            if only_cross_attention
            else attention_head_dim,
            processor=processor,
        )
        logger.debug(f"初始化attn1 (IPAttention): query_dim={dim}, heads={num_attention_heads}, dim_head={attention_head_dim}")
        
        # 2. 交叉注意力
        if cross_attention_dim is not None or double_self_attention:
            # 我们目前只在自注意力中使用AdaLayerNormZero，那里只会有一个注意力块
            # 即如果在第二个交叉注意力块中返回，AdaLayerZero返回的调制块数量将没有意义
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            logger.debug(f"初始化norm2: use_ada_layer_norm={self.use_ada_layer_norm}")

            # 初始化第二个注意力层（交叉注意力）
            self.attn2 = IPAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim
                if not double_self_attention
                else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                cross_attn_temporal_cond=ip_adapter_cross_attn,
                need_t2i_facein=need_t2i_facein,
                need_t2i_ip_adapter_face=need_t2i_ip_adapter_face,
                image_scale=image_scale,
                ip_adapter_dim=cross_attention_dim
                if not double_self_attention
                else attention_head_dim,
                facein_dim=cross_attention_dim
                if not double_self_attention
                else attention_head_dim,
                ip_adapter_face_dim=cross_attention_dim
                if not double_self_attention
                else attention_head_dim,
                processor=processor,
            )  # 如果encoder_hidden_states为None则是自注意力
            logger.debug(f"初始化attn2 (IPAttention): query_dim={dim}, cross_attention_dim={cross_attention_dim}")
        else:
            self.norm2 = None
            self.attn2 = None
            logger.debug("未初始化norm2和attn2")
            
        if self.attn1 is not None:
            if not allow_xformers:
                self.attn1.set_use_memory_efficient_attention_xformers = (
                    not_use_xformers_anyway
                )
                logger.debug("为attn1禁用xformers注意力机制")
        if self.attn2 is not None:
            if not allow_xformers:
                self.attn2.set_use_memory_efficient_attention_xformers = (
                    not_use_xformers_anyway
                )
                logger.debug("为attn2禁用xformers注意力机制")
                
        self.double_self_attention = double_self_attention
        self.only_cross_attention = only_cross_attention
        self.cross_attn_temporal_cond = cross_attn_temporal_cond
        self.image_scale = image_scale
        logger.debug(f"BasicTransformerBlock配置完成: double_self_attention={double_self_attention}, only_cross_attention={only_cross_attention}")

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        self_attn_block_embs: Optional[Tuple[List[torch.Tensor], List[None]]] = None,
        self_attn_block_embs_mode: Literal["read", "write"] = "write",
    ) -> torch.FloatTensor:
        """
        前向传播函数
        
        Args:
            hidden_states: 隐藏状态张量
            attention_mask: 注意力掩码
            encoder_hidden_states: 编码器隐藏状态
            encoder_attention_mask: 编码器注意力掩码
            timestep: 时间步长
            cross_attention_kwargs: 交叉注意力参数
            class_labels: 类别标签
            self_attn_block_embs: 自注意力块嵌入
            self_attn_block_embs_mode: 自注意力块嵌入模式("read"或"write")
            
        Returns:
            处理后的隐藏状态张量
        """
        logger.debug(f"BasicTransformerBlock前向传播开始: hidden_states.shape={hidden_states.shape}")
        
        # 注意，归一化总是在以下块中的实际计算之前应用
        # 0. 自注意力
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
            logger.debug(f"使用AdaLayerNorm进行归一化: norm_hidden_states.shape={norm_hidden_states.shape}")
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            logger.debug(f"使用AdaLayerNormZero进行归一化: norm_hidden_states.shape={norm_hidden_states.shape}")
        else:
            norm_hidden_states = self.norm1(hidden_states)
            logger.debug(f"使用LayerNorm进行归一化: norm_hidden_states.shape={norm_hidden_states.shape}")

        # 1. 获取lora缩放因子
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )
        logger.debug(f"获取lora_scale: {lora_scale}")

        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
            logger.debug("初始化cross_attention_kwargs为空字典")
            
        # 特殊AttnProcessor需要的入参在cross_attention_kwargs准备
        # special AttnProcessor needs input parameters in cross_attention_kwargs
        original_cross_attention_kwargs = {
            k: v
            for k, v in cross_attention_kwargs.items()
            if k
            not in [
                "num_frames",
                "sample_index",
                "vision_conditon_frames_sample_index",
                "vision_cond",
                "vision_clip_emb",
                "ip_adapter_scale",
                "face_emb",
                "facein_scale",
                "ip_adapter_face_emb",
                "ip_adapter_face_scale",
                "do_classifier_free_guidance",
            ]
        }
        logger.debug(f"提取original_cross_attention_kwargs，键值: {list(original_cross_attention_kwargs.keys())}")

        if "do_classifier_free_guidance" in cross_attention_kwargs:
            do_classifier_free_guidance = cross_attention_kwargs[
                "do_classifier_free_guidance"
            ]
            logger.debug(f"获取do_classifier_free_guidance: {do_classifier_free_guidance}")
        else:
            do_classifier_free_guidance = False
            logger.debug("do_classifier_free_guidance未设置，默认为False")

        # 2. 准备GLIGEN输入
        original_cross_attention_kwargs = (
            original_cross_attention_kwargs.copy()
            if original_cross_attention_kwargs is not None
            else {}
        )
        gligen_kwargs = original_cross_attention_kwargs.pop("gligen", None)
        logger.debug(f"处理GLIGEN参数: gligen_kwargs is not None = {gligen_kwargs is not None}")

        # 返回self_attn的结果，适用于referencenet的输出给其他Unet来使用
        # return the result of self_attn, which is suitable for the output of referencenet to be used by other Unet
        if (
            self_attn_block_embs is not None
            and self_attn_block_embs_mode.lower() == "write"
        ):
            # self_attn_block_emb = self.attn1.head_to_batch_dim(attn_output, out_dim=4)
            self_attn_block_emb = norm_hidden_states
            if not hasattr(self, "spatial_self_attn_idx"):
                raise ValueError(
                    "must call unet.insert_spatial_self_attn_idx to generate spatial attn index"
                )
            basick_transformer_idx = self.spatial_self_attn_idx
            if self.print_idx == 0:
                logger.debug(
                    f"self_attn_block_embs, self_attn_block_embs_mode={self_attn_block_embs_mode}, "
                    f"basick_transformer_idx={basick_transformer_idx}, length={len(self_attn_block_embs)}, shape={self_attn_block_emb.shape}, "
                )
            self_attn_block_embs[basick_transformer_idx] = self_attn_block_emb
            logger.debug(f"写入self_attn_block_embs: idx={basick_transformer_idx}, shape={self_attn_block_emb.shape}")

        # 读取并将referencenet嵌入放入cross_attention_kwargs中，这将被融合到attn_processor中
        # read and put referencenet emb into cross_attention_kwargs, which would be fused into attn_processor
        if (
            self_attn_block_embs is not None
            and self_attn_block_embs_mode.lower() == "read"
        ):
            basick_transformer_idx = self.spatial_self_attn_idx
            if not hasattr(self, "spatial_self_attn_idx"):
                raise ValueError(
                    "must call unet.insert_spatial_self_attn_idx to generate spatial attn index"
                )
            if self.print_idx == 0:
                logger.debug(
                    f"refer_self_attn_emb: , self_attn_block_embs_mode={self_attn_block_embs_mode}, "
                    f"length={len(self_attn_block_embs)}, idx={basick_transformer_idx}, "
                )
            ref_emb = self_attn_block_embs[basick_transformer_idx]
            cross_attention_kwargs["refer_emb"] = ref_emb
            if self.print_idx == 0:
                logger.debug(
                    f"unet attention read, {self.spatial_self_attn_idx}",
                )
                # ------------------------------warning-----------------------
                # 这两行由于使用了ref_emb会导致和checkpoint_train相关的训练错误，具体未知，留在这里作为警示
                # bellow annoated code will cause training error, keep it here as a warning
                # logger.debug(f"ref_emb shape,{ref_emb.shape}, {ref_emb.mean()}")
                # logger.debug(
                # f"norm_hidden_states shape, {norm_hidden_states.shape}, {norm_hidden_states.mean()}",
                # )
            logger.debug(f"读取self_attn_block_embs: idx={basick_transformer_idx}, shape={ref_emb.shape}")

        if self.attn1 is None:
            self.print_idx += 1
            logger.debug("attn1为None，直接返回norm_hidden_states")
            return norm_hidden_states
            
        # 执行第一个注意力层（自注意力）
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention
            else None,
            attention_mask=attention_mask,
            **(
                cross_attention_kwargs
                if isinstance(self.attn1.processor, BaseIPAttnProcessor)
                else original_cross_attention_kwargs
            ),
        )
        logger.debug(f"执行attn1完成: attn_output.shape={attn_output.shape}")

        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
            logger.debug("应用AdaLayerNormZero门控机制")
        hidden_states = attn_output + hidden_states
        logger.debug(f"自注意力输出与原始隐藏状态相加: hidden_states.shape={hidden_states.shape}")

        # 推断的时候，对于uncondition_部分独立生成，排除掉 refer_emb，
        # 首帧等的影响，避免生成参考了refer_emb、首帧等，又在uncond上去除了
        # in inference stage, eliminate influence of refer_emb, vis_cond on unconditionpart
        # to avoid use that, and then eliminate in pipeline
        # refer to moore-animate anyone

        # do_classifier_free_guidance = False
        if self.print_idx == 0:
            logger.debug(f"do_classifier_free_guidance={do_classifier_free_guidance},")
        if do_classifier_free_guidance:
            hidden_states_c = attn_output.clone()
            _uc_mask = (
                torch.Tensor(
                    [1] * (norm_hidden_states.shape[0] // 2)
                    + [0] * (norm_hidden_states.shape[0] // 2)
                )
                .to(norm_hidden_states.device)
                .bool()
            )
            hidden_states_c[_uc_mask] = self.attn1(
                norm_hidden_states[_uc_mask],
                encoder_hidden_states=norm_hidden_states[_uc_mask],
                attention_mask=attention_mask,
            )
            attn_output = hidden_states_c.clone()
            logger.debug("应用classifier-free guidance处理")

        if "refer_emb" in cross_attention_kwargs:
            del cross_attention_kwargs["refer_emb"]
            logger.debug("从cross_attention_kwargs中删除refer_emb")

        # 2.5 GLIGEN控制
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
            logger.debug("应用GLIGEN控制")
        # 2.5 结束

        # 3. 交叉注意力
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )
            logger.debug(f"对hidden_states进行第二次归一化: norm_hidden_states.shape={norm_hidden_states.shape}")

            # 特殊AttnProcessor需要的入参在cross_attention_kwargs准备
            # special AttnProcessor needs input parameters in cross_attention_kwargs
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states
                if not self.double_self_attention
                else None,
                attention_mask=encoder_attention_mask,
                **(
                    original_cross_attention_kwargs
                    if not isinstance(self.attn2.processor, BaseIPAttnProcessor)
                    else cross_attention_kwargs
                ),
            )
            logger.debug(f"执行attn2完成: attn_output.shape={attn_output.shape}")
            
            if self.print_idx == 0:
                logger.debug(
                    f"encoder_hidden_states, type={type(encoder_hidden_states)}"
                )
                if encoder_hidden_states is not None:
                    logger.debug(
                        f"encoder_hidden_states, ={encoder_hidden_states.shape}"
                    )

            # encoder_hidden_states_tmp = (
            #     encoder_hidden_states
            #     if not self.double_self_attention
            #     else norm_hidden_states
            # )
            # if do_classifier_free_guidance:
            #     hidden_states_c = attn_output.clone()
            #     _uc_mask = (
            #         torch.Tensor(
            #             [1] * (norm_hidden_states.shape[0] // 2)
            #             + [0] * (norm_hidden_states.shape[0] // 2)
            #         )
            #         .to(norm_hidden_states.device)
            #         .bool()
            #     )
            #     hidden_states_c[_uc_mask] = self.attn2(
            #         norm_hidden_states[_uc_mask],
            #         encoder_hidden_states=encoder_hidden_states_tmp[_uc_mask],
            #         attention_mask=attention_mask,
            #     )
            #     attn_output = hidden_states_c.clone()
            hidden_states = attn_output + hidden_states
            logger.debug(f"交叉注意力输出与隐藏状态相加: hidden_states.shape={hidden_states.shape}")
            
        # 4. 前馈网络
        if self.norm3 is not None and self.ff is not None:
            norm_hidden_states = self.norm3(hidden_states)
            logger.debug(f"对hidden_states进行第三次归一化: norm_hidden_states.shape={norm_hidden_states.shape}")
            
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                    norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )
                logger.debug("应用AdaLayerNormZero变换")
                
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" 可以用来节省内存
                if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                    raise ValueError(
                        f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                    )

                num_chunks = (
                    norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
                )
                ff_output = torch.cat(
                    [
                        self.ff(hid_slice, scale=lora_scale)
                        for hid_slice in norm_hidden_states.chunk(
                            num_chunks, dim=self._chunk_dim
                        )
                    ],
                    dim=self._chunk_dim,
                )
                logger.debug(f"分块执行前馈网络: num_chunks={num_chunks}")
            else:
                ff_output = self.ff(norm_hidden_states, scale=lora_scale)
                logger.debug("执行前馈网络")

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
                logger.debug("应用前馈网络门控机制")

            hidden_states = ff_output + hidden_states
            logger.debug(f"前馈网络输出与隐藏状态相加: hidden_states.shape={hidden_states.shape}")
            
        self.print_idx += 1
        logger.debug(f"BasicTransformerBlock前向传播完成: hidden_states.shape={hidden_states.shape}")
        return hidden_states