# Copyright 2023 The HuggingFace Team. All rights reserved.
# `TemporalConvLayer` Copyright 2023 Alibaba DAMO-VILAB, The ModelScope Team and The HuggingFace Team. All rights reserved.
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

# Adapted from https://github.com/huggingface/diffusers/blob/v0.16.1/src/diffusers/models/resnet.py
from __future__ import annotations

from functools import partial
from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers.models.resnet import TemporalConvLayer as DiffusersTemporalConvLayer
from ..data.data_util import batch_index_fill, batch_index_select
from . import Model_Register

# 设置日志记录器
logger = logging.getLogger(__name__)

@Model_Register.register
class TemporalConvLayer(nn.Module):
    """
    用于视频（图像序列）输入的时间卷积层
    代码主要复制自:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016
    """

    def __init__(
        self,
        in_dim,
        out_dim=None,
        dropout=0.0,
        keep_content_condition: bool = False,
        femb_channels: Optional[int] = None,
        need_temporal_weight: bool = True,
    ):
        """
        初始化时间卷积层
        
        Args:
            in_dim: 输入特征维度
            out_dim: 输出特征维度，默认与输入维度相同
            dropout: Dropout比率
            keep_content_condition: 是否保持内容条件帧
            femb_channels: 帧嵌入通道数
            need_temporal_weight: 是否需要时间权重
        """
        super().__init__()
        logger.debug(f"初始化TemporalConvLayer: in_dim={in_dim}, out_dim={out_dim}, dropout={dropout}")
        
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.keep_content_condition = keep_content_condition
        self.femb_channels = femb_channels
        self.need_temporal_weight = need_temporal_weight
        
        # 第一个卷积块：GroupNorm + SiLU激活 + 3D卷积 (3x1x1卷积核)
        logger.debug("创建第一个卷积块")
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim),  # 32组归一化
            nn.SiLU(),  # SiLU激活函数
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)),  # 时间维度3x1x1卷积
        )
        
        # 第二个卷积块：GroupNorm + SiLU激活 + Dropout + 3D卷积
        logger.debug("创建第二个卷积块")
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim),  # 32组归一化
            nn.SiLU(),  # SiLU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),  # 时间维度3x1x1卷积
        )
        
        # 第三个卷积块：GroupNorm + SiLU激活 + Dropout + 3D卷积
        logger.debug("创建第三个卷积块")
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_dim),  # 32组归一化
            nn.SiLU(),  # SiLU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),  # 时间维度3x1x1卷积
        )
        
        # 第四个卷积块：GroupNorm + SiLU激活 + Dropout + 3D卷积
        logger.debug("创建第四个卷积块")
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_dim),  # 32组归一化
            nn.SiLU(),  # SiLU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),  # 时间维度3x1x1卷积
        )

        # 时间权重参数，初始化为接近0的值
        logger.debug("初始化时间权重参数")
        self.temporal_weight = nn.Parameter(
            torch.tensor(
                [
                    1e-5,
                ]
            )
        )  # 初始化参数为0
        
        # 将最后一层参数置零，使卷积块初始时为恒等映射
        logger.debug("将最后一层卷积参数置零")
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)
        
        # 是否跳过时间层的标志
        self.skip_temporal_layers = False
        logger.debug("TemporalConvLayer初始化完成")

    def forward(
        self,
        hidden_states,
        num_frames=1,
        sample_index: torch.LongTensor = None,
        vision_conditon_frames_sample_index: torch.LongTensor = None,
        femb: torch.Tensor = None,
    ):
        """
        前向传播函数
        
        Args:
            hidden_states: 隐藏状态张量
            num_frames: 帧数
            sample_index: 采样索引
            vision_conditon_frames_sample_index: 视觉条件帧采样索引
            femb: 帧嵌入张量
            
        Returns:
            处理后的隐藏状态张量
        """
        logger.debug(f"TemporalConvLayer前向传播开始: hidden_states.shape={hidden_states.shape}, num_frames={num_frames}")
        
        # 如果设置了跳过时间层，则直接返回输入
        if self.skip_temporal_layers is True:
            logger.debug("跳过时间层，直接返回输入")
            return hidden_states
            
        # 保存输入数据类型
        hidden_states_dtype = hidden_states.dtype
        logger.debug(f"输入数据类型: {hidden_states_dtype}")
        
        # 重新排列张量形状: (b*t, c, h, w) -> (b, c, t, h, w)
        logger.debug("重新排列输入张量形状")
        hidden_states = rearrange(
            hidden_states, "(b t) c h w -> b c t h w", t=num_frames
        )
        logger.debug(f"重排后形状: {hidden_states.shape}")
        
        # 保存恒等映射用于残差连接
        identity = hidden_states
        logger.debug(f"恒等映射形状: {identity.shape}")
        
        # 依次通过四个卷积块
        logger.debug("通过第一个卷积块")
        hidden_states = self.conv1(hidden_states)
        logger.debug(f"conv1后形状: {hidden_states.shape}")
        
        logger.debug("通过第二个卷积块")
        hidden_states = self.conv2(hidden_states)
        logger.debug(f"conv2后形状: {hidden_states.shape}")
        
        logger.debug("通过第三个卷积块")
        hidden_states = self.conv3(hidden_states)
        logger.debug(f"conv3后形状: {hidden_states.shape}")
        
        logger.debug("通过第四个卷积块")
        hidden_states = self.conv4(hidden_states)
        logger.debug(f"conv4后形状: {hidden_states.shape}")
        
        # 如果需要保持内容条件帧，则进行特殊处理
        if self.keep_content_condition:
            logger.debug("处理内容条件帧保持逻辑")
            # 创建掩码张量
            mask = torch.ones_like(hidden_states, device=hidden_states.device)
            logger.debug(f"创建掩码张量，形状: {mask.shape}")
            
            # 使用batch_index_fill填充掩码
            logger.debug("使用batch_index_fill填充掩码")
            mask = batch_index_fill(
                mask, dim=2, index=vision_conditon_frames_sample_index, value=0
            )
            logger.debug(f"填充后掩码形状: {mask.shape}")
            
            # 根据是否需要时间权重应用不同的计算方式
            if self.need_temporal_weight:
                logger.debug("应用带时间权重的残差连接")
                hidden_states = (
                    identity + torch.abs(self.temporal_weight) * mask * hidden_states
                )
            else:
                logger.debug("应用不带时间权重的残差连接")
                hidden_states = identity + mask * hidden_states
        else:
            # 标准残差连接处理
            if self.need_temporal_weight:
                logger.debug("应用标准带时间权重的残差连接")
                hidden_states = (
                    identity + torch.abs(self.temporal_weight) * hidden_states
                )
            else:
                logger.debug("应用标准不带时间权重的残差连接")
                hidden_states = identity + hidden_states
                
        logger.debug(f"残差连接后形状: {hidden_states.shape}")
        
        # 重新排列回原始形状: (b, c, t, h, w) -> (b*t, c, h, w)
        logger.debug("重新排列回原始张量形状")
        hidden_states = rearrange(hidden_states, " b c t h w -> (b t) c h w")
        logger.debug(f"最终输出形状: {hidden_states.shape}")
        
        # 恢复原始数据类型
        hidden_states = hidden_states.to(dtype=hidden_states_dtype)
        logger.debug(f"恢复数据类型: {hidden_states_dtype}")
        logger.debug("TemporalConvLayer前向传播完成")
        
        return hidden_states