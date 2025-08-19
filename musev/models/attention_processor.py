# Copyright 2024 The HuggingFace Team. All rights reserved.
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

"""该模型是自定义的attn_processor，实现特殊功能的 Attn功能。
    相对而言，开源代码经常会重新定义Attention 类，
    
    This module implements special AttnProcessor function with custom attn_processor class.
    While other open source code always modify Attention class.
"""
# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
from __future__ import annotations

import time
from typing import Any, Callable, Optional
import logging

from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers
from diffusers.models.lora import LoRACompatibleLinear

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import (
    Attention as DiffusersAttention,
    AttnProcessor,
    AttnProcessor2_0,
)
from ..data.data_util import (
    batch_concat_two_tensor_with_index,
    batch_index_select,
    align_repeat_tensor_single_dim,
    batch_adain_conditioned_tensor,
)

from . import Model_Register

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.basicConfig(level=logging.info)

@maybe_allow_in_graph
class IPAttention(DiffusersAttention):
    r"""
    Modified Attention class which has special layer, like ip_apadapter_to_k, ip_apadapter_to_v,
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: str | None = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: int | None = None,
        norm_num_groups: int | None = None,
        spatial_norm_dim: int | None = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 0.00001,
        rescale_output_factor: float = 1,
        residual_connection: bool = False,
        _from_deprecated_attn_block=False,
        processor: AttnProcessor | None = None,
        cross_attn_temporal_cond: bool = False,
        image_scale: float = 1.0,
        ip_adapter_dim: int = None,
        need_t2i_facein: bool = False,
        facein_dim: int = None,
        need_t2i_ip_adapter_face: bool = False,
        ip_adapter_face_dim: int = None,
    ):
        """
        初始化IPAttention类
        
        Args:
            query_dim (int): 查询维度
            cross_attention_dim (int | None): 交叉注意力维度
            heads (int): 注意力头数
            dim_head (int): 每个注意力头的维度
            dropout (float): dropout比例
            bias (bool): 是否使用偏置
            upcast_attention (bool): 是否提升注意力计算精度
            upcast_softmax (bool): 是否提升softmax计算精度
            cross_attention_norm (str | None): 交叉注意力归一化方式
            cross_attention_norm_num_groups (int): 交叉注意力归一化组数
            added_kv_proj_dim (int | None): 添加的kv投影维度
            norm_num_groups (int | None): 归一化组数
            spatial_norm_dim (int | None): 空间归一化维度
            out_bias (bool): 输出是否使用偏置
            scale_qk (bool): 是否缩放qk
            only_cross_attention (bool): 是否仅使用交叉注意力
            eps (float): epsilon值
            rescale_output_factor (float): 输出缩放因子
            residual_connection (bool): 是否使用残差连接
            _from_deprecated_attn_block (bool): 是否来自废弃的注意力块
            processor (AttnProcessor | None): 注意力处理器
            cross_attn_temporal_cond (bool): 是否使用交叉注意力时间条件
            image_scale (float): 图像缩放因子
            ip_adapter_dim (int): IP适配器维度
            need_t2i_facein (bool): 是否需要T2I FaceIn
            facein_dim (int): FaceIn维度
            need_t2i_ip_adapter_face (bool): 是否需要T2I IP适配器面部
            ip_adapter_face_dim (int): IP适配器面部维度
        """
        super().__init__(
            query_dim,
            cross_attention_dim,
            heads,
            dim_head,
            dropout,
            bias,
            upcast_attention,
            upcast_softmax,
            cross_attention_norm,
            cross_attention_norm_num_groups,
            added_kv_proj_dim,
            norm_num_groups,
            spatial_norm_dim,
            out_bias,
            scale_qk,
            only_cross_attention,
            eps,
            rescale_output_factor,
            residual_connection,
            _from_deprecated_attn_block,
            processor,
        )
        self.cross_attn_temporal_cond = cross_attn_temporal_cond
        self.image_scale = image_scale
        # 面向首帧的 ip_adapter
        # ip_apdater
        if cross_attn_temporal_cond:
            # 初始化IP适配器的K和V投影层
            self.to_k_ip = LoRACompatibleLinear(ip_adapter_dim, query_dim, bias=False)
            self.to_v_ip = LoRACompatibleLinear(ip_adapter_dim, query_dim, bias=False)
        # facein
        self.need_t2i_facein = need_t2i_facein
        self.facein_dim = facein_dim
        if need_t2i_facein:
            raise NotImplementedError("facein")

        # ip_adapter_face
        self.need_t2i_ip_adapter_face = need_t2i_ip_adapter_face
        self.ip_adapter_face_dim = ip_adapter_face_dim
        if need_t2i_ip_adapter_face:
            # 初始化IP适配器面部的K和V投影层
            self.ip_adapter_face_to_k_ip = LoRACompatibleLinear(
                ip_adapter_face_dim, query_dim, bias=False
            )
            self.ip_adapter_face_to_v_ip = LoRACompatibleLinear(
                ip_adapter_face_dim, query_dim, bias=False
            )

    def set_use_memory_efficient_attention_xformers(
        self,
        use_memory_efficient_attention_xformers: bool,
        attention_op: Callable[..., Any] | None = None,
    ):
        """
        设置是否使用内存高效的xformers注意力
        
        Args:
            use_memory_efficient_attention_xformers (bool): 是否使用内存高效的xformers注意力
            attention_op (Callable[..., Any] | None): 注意力操作
        """
        if (
            "XFormers" in self.processor.__class__.__name__
            or "IP" in self.processor.__class__.__name__
        ):
            # 如果已经是XFormers或IP处理器，则跳过
            pass
        else:
            # 调用父类方法设置xformers注意力
            logger.debug("调用父类set_use_memory_efficient_attention_xformers方法")
            return super().set_use_memory_efficient_attention_xformers(
                use_memory_efficient_attention_xformers, attention_op
            )


@Model_Register.register
class BaseIPAttnProcessor(nn.Module):
    print_idx = 0

    def __init__(self, *args, **kwargs) -> None:
        """
        初始化BaseIPAttnProcessor类
        """
        super().__init__(*args, **kwargs)


@Model_Register.register
class T2IReferencenetIPAdapterXFormersAttnProcessor(BaseIPAttnProcessor):
    r"""
    面向 ref_image的 self_attn的 IPAdapter
    """
    print_idx = 0

    def __init__(
        self,
        attention_op: Optional[Callable] = None,
    ):
        """
        初始化T2IReferencenetIPAdapterXFormersAttnProcessor类
        
        Args:
            attention_op (Optional[Callable]): 注意力操作
        """
        super().__init__()

        self.attention_op = attention_op

    def __call__(
        self,
        attn: IPAttention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        num_frames: int = None,
        sample_index: torch.LongTensor = None,
        vision_conditon_frames_sample_index: torch.LongTensor = None,
        refer_emb: torch.Tensor = None,
        vision_clip_emb: torch.Tensor = None,
        ip_adapter_scale: float = 1.0,
        face_emb: torch.Tensor = None,
        facein_scale: float = 1.0,
        ip_adapter_face_emb: torch.Tensor = None,
        ip_adapter_face_scale: float = 1.0,
        do_classifier_free_guidance: bool = False,
    ):
        """
        执行注意力计算
        
        Args:
            attn (IPAttention): 注意力模块
            hidden_states (torch.FloatTensor): 隐藏状态
            encoder_hidden_states (Optional[torch.FloatTensor]): 编码器隐藏状态
            attention_mask (Optional[torch.FloatTensor]): 注意力掩码
            temb (Optional[torch.FloatTensor]): 时间嵌入
            scale (float): 缩放因子
            num_frames (int): 帧数
            sample_index (torch.LongTensor): 采样索引
            vision_conditon_frames_sample_index (torch.LongTensor): 视觉条件帧采样索引
            refer_emb (torch.Tensor): 参考嵌入
            vision_clip_emb (torch.Tensor): 视觉CLIP嵌入
            ip_adapter_scale (float): IP适配器缩放因子
            face_emb (torch.Tensor): 面部嵌入
            facein_scale (float): FaceIn缩放因子
            ip_adapter_face_emb (torch.Tensor): IP适配器面部嵌入
            ip_adapter_face_scale (float): IP适配器面部缩放因子
            do_classifier_free_guidance (bool): 是否执行分类器自由引导
            
        Returns:
            torch.FloatTensor: 处理后的隐藏状态
        """
        logger.debug(f"T2IReferencenetIPAdapterXFormersAttnProcessor.__call__开始执行")
        residual = hidden_states

        if attn.spatial_norm is not None:
            # 应用空间归一化
            logger.debug("应用spatial_norm")
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            # 将4D张量转换为3D张量
            batch_size, channel, height, width = hidden_states.shape
            logger.debug(f"将4D张量转换为3D张量: {hidden_states.shape} -> ({batch_size}, {channel}, {height * width})")
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        # 准备注意力掩码
        attention_mask = attn.prepare_attention_mask(
            attention_mask, key_tokens, batch_size
        )
        logger.debug(f"准备注意力掩码: {attention_mask.shape if attention_mask is not None else None}")
        
        if attention_mask is not None:
            # 扩展掩码的单例查询令牌维度
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            # 应用组归一化
            logger.debug("应用group_norm")
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        # 计算查询向量
        query = attn.to_q(hidden_states, scale=scale)
        logger.debug(f"计算查询向量: {query.shape}")

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            # 对编码器隐藏状态进行归一化
            logger.debug("对编码器隐藏状态进行归一化")
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        
        # 对齐并重复张量
        encoder_hidden_states = align_repeat_tensor_single_dim(
            encoder_hidden_states, target_length=hidden_states.shape[0], dim=0
        )
        logger.debug(f"对齐并重复张量后encoder_hidden_states: {encoder_hidden_states.shape}")
        
        # 计算键和值向量
        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)
        logger.debug(f"计算键向量: {key.shape}")
        logger.debug(f"计算值向量: {value.shape}")

        # for facein
        if self.print_idx == 0:
            logger.debug(
                f"T2IReferencenetIPAdapterXFormersAttnProcessor,type(face_emb)={type(face_emb)}, facein_scale={facein_scale}"
            )
        if facein_scale > 0 and face_emb is not None:
            raise NotImplementedError("facein")

        # 将头维度转换为批次维度
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        logger.debug(f"转换头维度到批次维度后 - query: {query.shape}, key: {key.shape}, value: {value.shape}")
        
        # 使用xformers进行内存高效注意力计算
        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            op=self.attention_op,
            scale=attn.scale,
        )
        logger.debug(f"基础注意力计算完成: {hidden_states.shape}")

        # ip-adapter start
        if self.print_idx == 0:
            logger.debug(
                f"T2IReferencenetIPAdapterXFormersAttnProcessor,type(vision_clip_emb)={type(vision_clip_emb)}"
            )
        if ip_adapter_scale > 0 and vision_clip_emb is not None:
            if self.print_idx == 0:
                logger.debug(
                    f"T2I cross_attn, ipadapter, vision_clip_emb={vision_clip_emb.shape}, hidden_states={hidden_states.shape}, batch_size={batch_size}"
                )
            # 计算IP适配器的键和值
            ip_key = attn.to_k_ip(vision_clip_emb)
            ip_value = attn.to_v_ip(vision_clip_emb)
            logger.debug(f"计算IP适配器的键和值 - ip_key: {ip_key.shape}, ip_value: {ip_value.shape}")
            
            # 对齐并重复张量
            ip_key = align_repeat_tensor_single_dim(
                ip_key, target_length=batch_size, dim=0
            )
            ip_value = align_repeat_tensor_single_dim(
                ip_value, target_length=batch_size, dim=0
            )
            logger.debug(f"对齐并重复IP适配器张量后 - ip_key: {ip_key.shape}, ip_value: {ip_value.shape}")
            
            # 转换头维度到批次维度
            ip_key = attn.head_to_batch_dim(ip_key).contiguous()
            ip_value = attn.head_to_batch_dim(ip_value).contiguous()
            if self.print_idx == 0:
                logger.debug(
                    f"query={query.shape}, ip_key={ip_key.shape}, ip_value={ip_value.shape}"
                )
            # 使用xformers进行IP适配器注意力计算
            hidden_states_from_ip = xformers.ops.memory_efficient_attention(
                query,
                ip_key,
                ip_value,
                attn_bias=attention_mask,
                op=self.attention_op,
                scale=attn.scale,
            )
            logger.debug(f"IP适配器注意力计算完成: {hidden_states_from_ip.shape}")
            # 将IP适配器结果与基础注意力结果融合
            hidden_states = hidden_states + ip_adapter_scale * hidden_states_from_ip
            logger.debug(f"融合IP适配器结果后: {hidden_states.shape}")
        # ip-adapter end

        # ip-adapter face start
        if self.print_idx == 0:
            logger.debug(
                f"T2IReferencenetIPAdapterXFormersAttnProcessor,type(ip_adapter_face_emb)={type(ip_adapter_face_emb)}"
            )
        if ip_adapter_face_scale > 0 and ip_adapter_face_emb is not None:
            if self.print_idx == 0:
                logger.debug(
                    f"T2I cross_attn, ipadapter face, ip_adapter_face_emb={vision_clip_emb.shape}, hidden_states={hidden_states.shape}, batch_size={batch_size}"
                )
            # 计算IP适配器面部的键和值
            ip_key = attn.ip_adapter_face_to_k_ip(ip_adapter_face_emb)
            ip_value = attn.ip_adapter_face_to_v_ip(ip_adapter_face_emb)
            logger.debug(f"计算IP适配器面部的键和值 - ip_key: {ip_key.shape}, ip_value: {ip_value.shape}")
            
            # 对齐并重复张量
            ip_key = align_repeat_tensor_single_dim(
                ip_key, target_length=batch_size, dim=0
            )
            ip_value = align_repeat_tensor_single_dim(
                ip_value, target_length=batch_size, dim=0
            )
            logger.debug(f"对齐并重复IP适配器面部张量后 - ip_key: {ip_key.shape}, ip_value: {ip_value.shape}")
            
            # 转换头维度到批次维度
            ip_key = attn.head_to_batch_dim(ip_key).contiguous()
            ip_value = attn.head_to_batch_dim(ip_value).contiguous()
            if self.print_idx == 0:
                logger.debug(
                    f"query={query.shape}, ip_key={ip_key.shape}, ip_value={ip_value.shape}"
                )
            # 使用xformers进行IP适配器面部注意力计算
            hidden_states_from_ip = xformers.ops.memory_efficient_attention(
                query,
                ip_key,
                ip_value,
                attn_bias=attention_mask,
                op=self.attention_op,
                scale=attn.scale,
            )
            logger.debug(f"IP适配器面部注意力计算完成: {hidden_states_from_ip.shape}")
            # 将IP适配器面部结果与基础注意力结果融合
            hidden_states = (
                hidden_states + ip_adapter_face_scale * hidden_states_from_ip
            )
            logger.debug(f"融合IP适配器面部结果后: {hidden_states.shape}")
        # ip-adapter face end

        # 转换数据类型并恢复头维度
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        logger.debug(f"转换数据类型并恢复头维度后: {hidden_states.shape}")

        # 线性投影
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        logger.debug(f"线性投影后: {hidden_states.shape}")
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        logger.debug(f"应用dropout后: {hidden_states.shape}")

        if input_ndim == 4:
            # 将3D张量转换回4D张量
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
            logger.debug(f"将3D张量转换回4D张量: {hidden_states.shape}")

        if attn.residual_connection:
            # 应用残差连接
            hidden_states = hidden_states + residual
            logger.debug("应用残差连接")

        hidden_states = hidden_states / attn.rescale_output_factor
        self.print_idx += 1
        logger.debug(f"T2IReferencenetIPAdapterXFormersAttnProcessor.__call__执行完成")
        return hidden_states


@Model_Register.register
class NonParamT2ISelfReferenceXFormersAttnProcessor(BaseIPAttnProcessor):
    r"""
    面向首帧的 referenceonly attn,适用于 T2I的 self_attn
    referenceonly with vis_cond as key, value, in t2i self_attn.
    """
    print_idx = 0

    def __init__(
        self,
        attention_op: Optional[Callable] = None,
    ):
        """
        初始化NonParamT2ISelfReferenceXFormersAttnProcessor类
        
        Args:
            attention_op (Optional[Callable]): 注意力操作
        """
        super().__init__()

        self.attention_op = attention_op

    def __call__(
        self,
        attn: IPAttention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        num_frames: int = None,
        sample_index: torch.LongTensor = None,
        vision_conditon_frames_sample_index: torch.LongTensor = None,
        refer_emb: torch.Tensor = None,
        face_emb: torch.Tensor = None,
        vision_clip_emb: torch.Tensor = None,
        ip_adapter_scale: float = 1.0,
        facein_scale: float = 1.0,
        ip_adapter_face_emb: torch.Tensor = None,
        ip_adapter_face_scale: float = 1.0,
        do_classifier_free_guidance: bool = False,
    ):
        """
        执行注意力计算
        
        Args:
            attn (IPAttention): 注意力模块
            hidden_states (torch.FloatTensor): 隐藏状态
            encoder_hidden_states (Optional[torch.FloatTensor]): 编码器隐藏状态
            attention_mask (Optional[torch.FloatTensor]): 注意力掩码
            temb (Optional[torch.FloatTensor]): 时间嵌入
            scale (float): 缩放因子
            num_frames (int): 帧数
            sample_index (torch.LongTensor): 采样索引
            vision_conditon_frames_sample_index (torch.LongTensor): 视觉条件帧采样索引
            refer_emb (torch.Tensor): 参考嵌入
            face_emb (torch.Tensor): 面部嵌入
            vision_clip_emb (torch.Tensor): 视觉CLIP嵌入
            ip_adapter_scale (float): IP适配器缩放因子
            facein_scale (float): FaceIn缩放因子
            ip_adapter_face_emb (torch.Tensor): IP适配器面部嵌入
            ip_adapter_face_scale (float): IP适配器面部缩放因子
            do_classifier_free_guidance (bool): 是否执行分类器自由引导
            
        Returns:
            torch.FloatTensor: 处理后的隐藏状态
        """
        logger.debug(f"NonParamT2ISelfReferenceXFormersAttnProcessor.__call__开始执行")
        residual = hidden_states

        if attn.spatial_norm is not None:
            # 应用空间归一化
            logger.debug("应用spatial_norm")
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            # 将4D张量转换为3D张量
            batch_size, channel, height, width = hidden_states.shape
            logger.debug(f"将4D张量转换为3D张量: {hidden_states.shape} -> ({batch_size}, {channel}, {height * width})")
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        # 准备注意力掩码
        attention_mask = attn.prepare_attention_mask(
            attention_mask, key_tokens, batch_size
        )
        logger.debug(f"准备注意力掩码: {attention_mask.shape if attention_mask is not None else None}")
        
        if attention_mask is not None:
            # 扩展掩码的单例查询令牌维度
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        # vision_cond in same unet attn start
        if (
            vision_conditon_frames_sample_index is not None and num_frames > 1
        ) or refer_emb is not None:
            batchsize_timesize = hidden_states.shape[0]
            if self.print_idx == 0:
                logger.debug(
                    f"NonParamT2ISelfReferenceXFormersAttnProcessor 0, hidden_states={hidden_states.shape}, vision_conditon_frames_sample_index={vision_conditon_frames_sample_index}"
                )
            # 重新排列隐藏状态
            encoder_hidden_states = rearrange(
                hidden_states, "(b t) hw c -> b t hw c", t=num_frames
            )
            logger.debug(f"重新排列隐藏状态: {hidden_states.shape} -> {encoder_hidden_states.shape}")
            
            if vision_conditon_frames_sample_index is not None and num_frames > 1:
                # 根据索引选择视觉条件帧
                ip_hidden_states = batch_index_select(
                    encoder_hidden_states,
                    dim=1,
                    index=vision_conditon_frames_sample_index,
                ).contiguous()
                logger.debug(f"根据索引选择视觉条件帧: {encoder_hidden_states.shape} -> {ip_hidden_states.shape}")
                
                if self.print_idx == 0:
                    logger.debug(
                        f"NonParamT2ISelfReferenceXFormersAttnProcessor 1, vis_cond referenceonly, encoder_hidden_states={encoder_hidden_states.shape}, ip_hidden_states={ip_hidden_states.shape}"
                    )
                # 重新排列IP隐藏状态
                ip_hidden_states = rearrange(
                    ip_hidden_states, "b t hw c -> b 1 (t hw) c"
                )
                # 对齐并重复张量
                ip_hidden_states = align_repeat_tensor_single_dim(
                    ip_hidden_states,
                    dim=1,
                    target_length=num_frames,
                )
                logger.debug(f"对齐并重复张量: {ip_hidden_states.shape}")
                
                if self.print_idx == 0:
                    logger.debug(
                        f"NonParamT2ISelfReferenceXFormersAttnProcessor 2, vis_cond referenceonly, encoder_hidden_states={encoder_hidden_states.shape}, ip_hidden_states={ip_hidden_states.shape}"
                    )
                # 连接编码器隐藏状态和IP隐藏状态
                encoder_hidden_states = torch.concat(
                    [encoder_hidden_states, ip_hidden_states], dim=2
                )
                logger.debug(f"连接编码器隐藏状态和IP隐藏状态: {encoder_hidden_states.shape}")
                
                if self.print_idx == 0:
                    logger.debug(
                        f"NonParamT2ISelfReferenceXFormersAttnProcessor 3, hidden_states={hidden_states.shape}, ip_hidden_states={ip_hidden_states.shape}"
                    )
                    
            if refer_emb is not None:  # and num_frames > 1:
                # 处理参考嵌入
                refer_emb = rearrange(refer_emb, "b c t h w->b 1 (t h w) c")
                refer_emb = align_repeat_tensor_single_dim(
                    refer_emb, target_length=num_frames, dim=1
                )
                logger.debug(f"处理参考嵌入: {refer_emb.shape}")
                
                if self.print_idx == 0:
                    logger.debug(
                        f"NonParamT2ISelfReferenceXFormersAttnProcessor4, referencenet, encoder_hidden_states={encoder_hidden_states.shape}, refer_emb={refer_emb.shape}"
                    )
                # 连接编码器隐藏状态和参考嵌入
                encoder_hidden_states = torch.concat(
                    [encoder_hidden_states, refer_emb], dim=2
                )
                logger.debug(f"连接编码器隐藏状态和参考嵌入: {encoder_hidden_states.shape}")
                
                if self.print_idx == 0:
                    logger.debug(
                        f"NonParamT2ISelfReferenceXFormersAttnProcessor5, referencenet, encoder_hidden_states={encoder_hidden_states.shape}, refer_emb={refer_emb.shape}"
                    )
            # 重新排列编码器隐藏状态
            encoder_hidden_states = rearrange(
                encoder_hidden_states, "b t hw c -> (b t) hw c"
            )
            logger.debug(f"重新排列编码器隐藏状态: {encoder_hidden_states.shape}")
        #  vision_cond in same unet attn end

        if attn.group_norm is not None:
            # 应用组归一化
            logger.debug("应用group_norm")
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        # 计算查询向量
        query = attn.to_q(hidden_states, scale=scale)
        logger.debug(f"计算查询向量: {query.shape}")

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            # 对编码器隐藏状态进行归一化
            logger.debug("对编码器隐藏状态进行归一化")
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        
        # 对齐并重复张量
        encoder_hidden_states = align_repeat_tensor_single_dim(
            encoder_hidden_states, target_length=hidden_states.shape[0], dim=0
        )
        logger.debug(f"对齐并重复张量后encoder_hidden_states: {encoder_hidden_states.shape}")
        
        # 计算键和值向量
        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)
        logger.debug(f"计算键向量: {key.shape}")
        logger.debug(f"计算值向量: {value.shape}")

        # 转换头维度到批次维度
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        logger.debug(f"转换头维度到批次维度后 - query: {query.shape}, key: {key.shape}, value: {value.shape}")

        # 使用xformers进行内存高效注意力计算
        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            op=self.attention_op,
            scale=attn.scale,
        )
        logger.debug(f"注意力计算完成: {hidden_states.shape}")
        
        # 转换数据类型并恢复头维度
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        logger.debug(f"转换数据类型并恢复头维度后: {hidden_states.shape}")

        # 线性投影
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        logger.debug(f"线性投影后: {hidden_states.shape}")
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        logger.debug(f"应用dropout后: {hidden_states.shape}")

        if input_ndim == 4:
            # 将3D张量转换回4D张量
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
            logger.debug(f"将3D张量转换回4D张量: {hidden_states.shape}")

        if attn.residual_connection:
            # 应用残差连接
            hidden_states = hidden_states + residual
            logger.debug("应用残差连接")

        hidden_states = hidden_states / attn.rescale_output_factor
        self.print_idx += 1
        logger.debug(f"NonParamT2ISelfReferenceXFormersAttnProcessor.__call__执行完成")

        return hidden_states


@Model_Register.register
class NonParamReferenceIPXFormersAttnProcessor(
    NonParamT2ISelfReferenceXFormersAttnProcessor
):
    def __init__(self, attention_op: Callable[..., Any] | None = None):
        """
        初始化NonParamReferenceIPXFormersAttnProcessor类
        
        Args:
            attention_op (Callable[..., Any] | None): 注意力操作
        """
        super().__init__(attention_op)


@maybe_allow_in_graph
class ReferEmbFuseAttention(IPAttention):
    """使用 attention 融合 refernet 中的 emb 到 unet 对应的 latens 中
    # TODO: 目前只支持 bt hw c 的融合，后续考虑增加对 视频 bhw t c、b thw c的融合
    residual_connection: bool = True, 默认， 从不产生影响开始学习

    use attention to fuse referencenet emb into unet latents
    # TODO: by now, only support bt hw c, later consider to support bhw t c, b thw c
    residual_connection: bool = True, default, start from no effect

    Args:
        IPAttention (_type_): _description_
    """

    print_idx = 0

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: str | None = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: int | None = None,
        norm_num_groups: int | None = None,
        spatial_norm_dim: int | None = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 0.00001,
        rescale_output_factor: float = 1,
        residual_connection: bool = True,
        _from_deprecated_attn_block=False,
        processor: AttnProcessor | None = None,
        cross_attn_temporal_cond: bool = False,
        image_scale: float = 1,
    ):
        """
        初始化ReferEmbFuseAttention类
        
        Args:
            query_dim (int): 查询维度
            cross_attention_dim (int | None): 交叉注意力维度
            heads (int): 注意力头数
            dim_head (int): 每个注意力头的维度
            dropout (float): dropout比例
            bias (bool): 是否使用偏置
            upcast_attention (bool): 是否提升注意力计算精度
            upcast_softmax (bool): 是否提升softmax计算精度
            cross_attention_norm (str | None): 交叉注意力归一化方式
            cross_attention_norm_num_groups (int): 交叉注意力归一化组数
            added_kv_proj_dim (int | None): 添加的kv投影维度
            norm_num_groups (int | None): 归一化组数
            spatial_norm_dim (int | None): 空间归一化维度
            out_bias (bool): 输出是否使用偏置
            scale_qk (bool): 是否缩放qk
            only_cross_attention (bool): 是否仅使用交叉注意力
            eps (float): epsilon值
            rescale_output_factor (float): 输出缩放因子
            residual_connection (bool): 是否使用残差连接
            _from_deprecated_attn_block (bool): 是否来自废弃的注意力块
            processor (AttnProcessor | None): 注意力处理器
            cross_attn_temporal_cond (bool): 是否使用交叉注意力时间条件
            image_scale (float): 图像缩放因子
        """
        super().__init__(
            query_dim,
            cross_attention_dim,
            heads,
            dim_head,
            dropout,
            bias,
            upcast_attention,
            upcast_softmax,
            cross_attention_norm,
            cross_attention_norm_num_groups,
            added_kv_proj_dim,
            norm_num_groups,
            spatial_norm_dim,
            out_bias,
            scale_qk,
            only_cross_attention,
            eps,
            rescale_output_factor,
            residual_connection,
            _from_deprecated_attn_block,
            processor,
            cross_attn_temporal_cond,
            image_scale,
        )
        self.processor = None
        # 配合residual,使一开始不影响之前结果
        nn.init.zeros_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        num_frames: int = None,
    ) -> torch.Tensor:
        """
        融合参考网络嵌入到UNet潜在空间中
        
        Args:
            hidden_states (torch.FloatTensor): UNet潜在状态，形状为 (b t1) c h1 w1
            encoder_hidden_states (Optional[torch.FloatTensor], optional): 参考网络嵌入，形状为 b c2 t2 h2 w2
            attention_mask (Optional[torch.FloatTensor], optional): 注意力掩码
            temb (Optional[torch.FloatTensor], optional): 时间嵌入
            scale (float, optional): 缩放因子
            num_frames (int, optional): 帧数
            
        Returns:
            torch.Tensor: 融合后的张量
        """
        logger.debug(f"ReferEmbFuseAttention.forward开始执行")
        residual = hidden_states
        # 开始处理
        hidden_states = rearrange(
            hidden_states, "(b t) c h w -> b c t h w", t=num_frames
        )
        logger.debug(f"重新排列hidden_states: {residual.shape} -> {hidden_states.shape}")
        
        batch_size, channel, t1, height, width = hidden_states.shape
        if self.print_idx == 0:
            logger.debug(
                f"hidden_states={hidden_states.shape},encoder_hidden_states={encoder_hidden_states.shape}"
            )
        # 在hw通道中将hidden_states b c t1 h1 w1与encoder_hidden_states连接，形成bt (t2 + 1)hw c
        encoder_hidden_states = rearrange(
            encoder_hidden_states, " b c t2 h w-> b (t2 h w) c"
        )
        logger.debug(f"重新排列encoder_hidden_states: {encoder_hidden_states.shape}")
        
        encoder_hidden_states = repeat(
            encoder_hidden_states, " b t2hw c -> (b t) t2hw c", t=t1
        )
        logger.debug(f"重复encoder_hidden_states: {encoder_hidden_states.shape}")
        
        hidden_states = rearrange(hidden_states, " b c t h w-> (b t) (h w) c")
        logger.debug(f"重新排列hidden_states: {hidden_states.shape}")
        
        # bt (t2+1)hw d
        encoder_hidden_states = torch.concat(
            [encoder_hidden_states, hidden_states], dim=1
        )
        logger.debug(f"连接encoder_hidden_states和hidden_states: {encoder_hidden_states.shape}")
        # end

        if self.spatial_norm is not None:
            # 应用空间归一化
            logger.debug("应用spatial_norm")
            hidden_states = self.spatial_norm(hidden_states, temb)

        _, key_tokens, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        # 准备注意力掩码
        attention_mask = self.prepare_attention_mask(
            attention_mask, key_tokens, batch_size
        )
        logger.debug(f"准备注意力掩码: {attention_mask.shape if attention_mask is not None else None}")
        
        if attention_mask is not None:
            # 扩展掩码的单例查询令牌维度
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if self.group_norm is not None:
            # 应用组归一化
            logger.debug("应用group_norm")
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        # 计算查询向量
        query = self.to_q(hidden_states, scale=scale)
        logger.debug(f"计算查询向量: {query.shape}")

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            # 对编码器隐藏状态进行归一化
            logger.debug("对编码器隐藏状态进行归一化")
            encoder_hidden_states = self.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        # 计算键和值向量
        key = self.to_k(encoder_hidden_states, scale=scale)
        value = self.to_v(encoder_hidden_states, scale=scale)
        logger.debug(f"计算键向量: {key.shape}")
        logger.debug(f"计算值向量: {value.shape}")

        # 转换头维度到批次维度
        query = self.head_to_batch_dim(query).contiguous()
        key = self.head_to_batch_dim(key).contiguous()
        value = self.head_to_batch_dim(value).contiguous()
        logger.debug(f"转换头维度到批次维度后 - query: {query.shape}, key: {key.shape}, value: {value.shape}")

        # query: b t hw d
        # key/value: bt (t1+1)hw d
        # 使用xformers进行内存高效注意力计算
        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            scale=self.scale,
        )
        logger.debug(f"注意力计算完成: {hidden_states.shape}")
        
        # 转换数据类型并恢复头维度
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.batch_to_head_dim(hidden_states)
        logger.debug(f"转换数据类型并恢复头维度后: {hidden_states.shape}")

        # 线性投影
        hidden_states = self.to_out[0](hidden_states, scale=scale)
        logger.debug(f"线性投影后: {hidden_states.shape}")
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        logger.debug(f"应用dropout后: {hidden_states.shape}")

        # 重新排列输出张量
        hidden_states = rearrange(
            hidden_states,
            "bt (h w) c-> bt c h w",
            h=height,
            w=width,
        )
        logger.debug(f"重新排列输出张量: {hidden_states.shape}")
        
        if self.residual_connection:
            # 应用残差连接
            hidden_states = hidden_states + residual
            logger.debug("应用残差连接")

        hidden_states = hidden_states / self.rescale_output_factor
        self.print_idx += 1
        logger.debug(f"ReferEmbFuseAttention.forward执行完成")
        return hidden_states