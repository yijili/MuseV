# -*- coding: utf-8 -*-
"""
IP-Adapter 加载器模块，用于加载和配置 IP-Adapter 相关组件
该模块提供了一系列函数来加载视觉编码器、图像投影模型等 IP-Adapter 组件
"""

import copy
from typing import Any, Callable, Dict, Iterable, Union
import PIL
import cv2
import torch
import argparse
import datetime
import logging
import inspect
import math
import os
import shutil
from typing import Dict, List, Optional, Tuple
from pprint import pprint
from collections import OrderedDict
from dataclasses import dataclass
import gc
import time

import numpy as np
from omegaconf import OmegaConf
from omegaconf import SCMode
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
import pandas as pd
import h5py
from diffusers.models.modeling_utils import load_state_dict
from diffusers.utils import (
    logging,
)
from diffusers.utils.import_utils import is_xformers_available

# 导入视觉特征提取器模块
from mmcm.vision.feature_extractor import clip_vision_extractor
from mmcm.vision.feature_extractor.clip_vision_extractor import (
    ImageClipVisionFeatureExtractor,
    ImageClipVisionFeatureExtractorV2,
    VerstailSDLastHiddenState2ImageEmb,
)

# 导入 IP-Adapter 的相关组件
from ip_adapter.resampler import Resampler
from ip_adapter.ip_adapter import ImageProjModel

# 导入 UNet 相关加载器
from .unet_loader import update_unet_with_sd
from .unet_3d_condition import UNet3DConditionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def load_vision_clip_encoder_by_name(
    ip_image_encoder: Tuple[str, nn.Module] = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    vision_clip_extractor_class_name: str = None,
) -> nn.Module:
    """
    根据名称加载视觉 CLIP 编码器
    
    Args:
        ip_image_encoder (Tuple[str, nn.Module], optional): IP 图像编码器路径或模型. Defaults to None.
        dtype (torch.dtype, optional): 模型数据类型. Defaults to torch.float16.
        device (str, optional): 运行设备. Defaults to "cuda".
        vision_clip_extractor_class_name (str, optional): 视觉 CLIP 提取器类名. Defaults to None.
        
    Returns:
        nn.Module: 加载的视觉 CLIP 编码器模型
    """
    logger.info(f"开始加载视觉 CLIP 编码器: {vision_clip_extractor_class_name}")
    
    if vision_clip_extractor_class_name is not None:
        # 根据类名动态获取并实例化对应的视觉特征提取器
        vision_clip_extractor = getattr(
            clip_vision_extractor, vision_clip_extractor_class_name
        )(
            pretrained_model_name_or_path=ip_image_encoder,
            dtype=dtype,
            device=device,
        )
        logger.info(f"成功加载视觉 CLIP 编码器类: {vision_clip_extractor_class_name}")
    else:
        vision_clip_extractor = None
        logger.info("未指定视觉 CLIP 编码器类名，返回 None")
        
    return vision_clip_extractor


def load_ip_adapter_image_proj_by_name(
    model_name: str,
    ip_ckpt: Tuple[str, nn.Module] = None,
    cross_attention_dim: int = 768,
    clip_embeddings_dim: int = 1024,
    clip_extra_context_tokens: int = 4,
    ip_scale: float = 0.0,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    unet: nn.Module = None,
    vision_clip_extractor_class_name: str = None,
    ip_image_encoder: Tuple[str, nn.Module] = None,
) -> nn.Module:
    """
    根据模型名称加载 IP-Adapter 图像投影模型
    
    Args:
        model_name (str): 模型名称
        ip_ckpt (Tuple[str, nn.Module], optional): IP 检查点路径或模型. Defaults to None.
        cross_attention_dim (int, optional): 交叉注意力维度. Defaults to 768.
        clip_embeddings_dim (int, optional): CLIP 嵌入维度. Defaults to 1024.
        clip_extra_context_tokens (int, optional): 额外上下文标记数. Defaults to 4.
        ip_scale (float, optional): IP 缩放因子. Defaults to 0.0.
        dtype (torch.dtype, optional): 模型数据类型. Defaults to torch.float16.
        device (str, optional): 运行设备. Defaults to "cuda".
        unet (nn.Module, optional): UNet 模型. Defaults to None.
        vision_clip_extractor_class_name (str, optional): 视觉 CLIP 提取器类名. Defaults to None.
        ip_image_encoder (Tuple[str, nn.Module], optional): IP 图像编码器. Defaults to None.
        
    Returns:
        nn.Module: 加载的 IP-Adapter 图像投影模型
    """
    logger.info(f"开始加载 IP-Adapter 图像投影模型: {model_name}")
    
    if model_name in [
        "IPAdapter",
        "musev_referencenet",
        "musev_referencenet_pose",
    ]:
        # 加载基础的 ImageProjModel
        ip_adapter_image_proj = ImageProjModel(
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=clip_extra_context_tokens,
        )
        logger.info(f"加载基础 ImageProjModel，模型名称: {model_name}")

    elif model_name == "IPAdapterPlus":
        # 加载增强版的 IP-Adapter Plus 模型
        vision_clip_extractor = ImageClipVisionFeatureExtractorV2(
            pretrained_model_name_or_path=ip_image_encoder,
            dtype=dtype,
            device=device,
        )
        ip_adapter_image_proj = Resampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=clip_extra_context_tokens,
            embedding_dim=vision_clip_extractor.image_encoder.config.hidden_size,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )
        logger.info("加载 IPAdapterPlus 模型")

    elif model_name in [
        "VerstailSDLastHiddenState2ImageEmb",
        "OriginLastHiddenState2ImageEmbd",
        "OriginLastHiddenState2Poolout",
    ]:
        # 加载特定类型的图像嵌入模型
        ip_adapter_image_proj = getattr(
            clip_vision_extractor, model_name
        ).from_pretrained(ip_image_encoder)
        logger.info(f"加载特定图像嵌入模型: {model_name}")
        
    else:
        # 不支持的模型类型
        error_msg = f"不支持的模型名称={model_name}，仅支持 IPAdapter, IPAdapterPlus, VerstailSDLastHiddenState2ImageEmb"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    # 如果提供了检查点，则加载模型权重
    if ip_ckpt is not None:
        logger.info(f"从检查点加载 IP-Adapter 权重: {ip_ckpt}")
        ip_adapter_state_dict = torch.load(
            ip_ckpt,
            map_location="cpu",
        )
        ip_adapter_image_proj.load_state_dict(ip_adapter_state_dict["image_proj"])
        logger.info("成功加载 image_proj 权重")
        
        # 如果 UNet 支持 IP-Adapter 交叉注意力并且状态字典中包含相关参数，则更新 UNet 参数
        if (
            unet is not None
            and unet.ip_adapter_cross_attn
            and "ip_adapter" in ip_adapter_state_dict
        ):
            update_unet_ip_adapter_cross_attn_param(
                unet, ip_adapter_state_dict["ip_adapter"]
            )
            logger.info(
                f"更新 unet.spatial_cross_attn_ip_adapter 参数，使用检查点: {ip_ckpt}"
            )
            
    logger.info(f"完成 IP-Adapter 图像投影模型加载: {model_name}")
    return ip_adapter_image_proj


def load_ip_adapter_vision_clip_encoder_by_name(
    model_name: str,
    ip_ckpt: Tuple[str, nn.Module],
    ip_image_encoder: Tuple[str, nn.Module] = None,
    cross_attention_dim: int = 768,
    clip_embeddings_dim: int = 1024,
    clip_extra_context_tokens: int = 4,
    ip_scale: float = 0.0,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    unet: nn.Module = None,
    vision_clip_extractor_class_name: str = None,
) -> nn.Module:
    """
    根据模型名称加载 IP-Adapter 视觉 CLIP 编码器和图像投影模型
    
    Args:
        model_name (str): 模型名称
        ip_ckpt (Tuple[str, nn.Module]): IP 检查点路径或模型
        ip_image_encoder (Tuple[str, nn.Module], optional): IP 图像编码器. Defaults to None.
        cross_attention_dim (int, optional): 交叉注意力维度. Defaults to 768.
        clip_embeddings_dim (int, optional): CLIP 嵌入维度. Defaults to 1024.
        clip_extra_context_tokens (int, optional): 额外上下文标记数. Defaults to 4.
        ip_scale (float, optional): IP 缩放因子. Defaults to 0.0.
        dtype (torch.dtype, optional): 模型数据类型. Defaults to torch.float16.
        device (str, optional): 运行设备. Defaults to "cuda".
        unet (nn.Module, optional): UNet 模型. Defaults to None.
        vision_clip_extractor_class_name (str, optional): 视觉 CLIP 提取器类名. Defaults to None.
        
    Returns:
        nn.Module: 加载的视觉 CLIP 编码器和图像投影模型元组
    """
    logger.info(f"开始加载 IP-Adapter 视觉 CLIP 编码器和图像投影模型: {model_name}")
    
    # 加载视觉 CLIP 编码器
    if vision_clip_extractor_class_name is not None:
        vision_clip_extractor = getattr(
            clip_vision_extractor, vision_clip_extractor_class_name
        )(
            pretrained_model_name_or_path=ip_image_encoder,
            dtype=dtype,
            device=device,
        )
        logger.info(f"加载视觉 CLIP 编码器类: {vision_clip_extractor_class_name}")
    else:
        vision_clip_extractor = None
        logger.info("未指定视觉 CLIP 编码器类名")
        
    # 根据模型名称加载对应的模型组件
    if model_name in [
        "IPAdapter",
        "musev_referencenet",
    ]:
        if ip_image_encoder is not None:
            if vision_clip_extractor_class_name is None:
                vision_clip_extractor = ImageClipVisionFeatureExtractor(
                    pretrained_model_name_or_path=ip_image_encoder,
                    dtype=dtype,
                    device=device,
                )
                logger.info("加载 ImageClipVisionFeatureExtractor")
        else:
            vision_clip_extractor = None
            logger.info("未提供图像编码器，设置视觉 CLIP 编码器为 None")
            
        # 加载基础的 ImageProjModel
        ip_adapter_image_proj = ImageProjModel(
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=clip_extra_context_tokens,
        )
        logger.info(f"加载基础 ImageProjModel，模型名称: {model_name}")

    elif model_name == "IPAdapterPlus":
        if ip_image_encoder is not None:
            if vision_clip_extractor_class_name is None:
                vision_clip_extractor = ImageClipVisionFeatureExtractorV2(
                    pretrained_model_name_or_path=ip_image_encoder,
                    dtype=dtype,
                    device=device,
                )
                logger.info("加载 ImageClipVisionFeatureExtractorV2")
        else:
            vision_clip_extractor = None
            logger.info("未提供图像编码器，设置视觉 CLIP 编码器为 None")
            
        # 加载 Resampler 模型
        ip_adapter_image_proj = Resampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=clip_extra_context_tokens,
            embedding_dim=vision_clip_extractor.image_encoder.config.hidden_size,
            output_dim=cross_attention_dim,
            ff_mult=4,
        ).to(dtype=torch.float16)
        logger.info("加载 Resampler 模型")

    else:
        # 不支持的模型类型
        error_msg = f"不支持的模型名称={model_name}，仅支持 IPAdapter, IPAdapterPlus"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    # 加载检查点并更新模型参数
    logger.info(f"从检查点加载 IP-Adapter 权重: {ip_ckpt}")
    ip_adapter_state_dict = torch.load(
        ip_ckpt,
        map_location="cpu",
    )
    ip_adapter_image_proj.load_state_dict(ip_adapter_state_dict["image_proj"])
    logger.info("成功加载 image_proj 权重")
    
    # 如果 UNet 支持 IP-Adapter 交叉注意力并且状态字典中包含相关参数，则更新 UNet 参数
    if (
        unet is not None
        and unet.ip_adapter_cross_attn
        and "ip_adapter" in ip_adapter_state_dict
    ):
        update_unet_ip_adapter_cross_attn_param(
            unet, ip_adapter_state_dict["ip_adapter"]
        )
        logger.info(
            f"更新 unet.spatial_cross_attn_ip_adapter 参数，使用检查点: {ip_ckpt}"
        )
        
    logger.info(f"完成 IP-Adapter 视觉 CLIP 编码器和图像投影模型加载: {model_name}")
    return (
        vision_clip_extractor,
        ip_adapter_image_proj,
    )


# 参考 https://github.com/tencent-ailab/IP-Adapter/issues/168#issuecomment-1846771651
# 定义 UNet 中的 IP-Adapter 相关键列表
unet_keys_list = [
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.to_v_ip.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight",
]

# 定义 IP-Adapter 中的键列表
ip_adapter_keys_list = [
    "1.to_k_ip.weight",
    "1.to_v_ip.weight",
    "3.to_k_ip.weight",
    "3.to_v_ip.weight",
    "5.to_k_ip.weight",
    "5.to_v_ip.weight",
    "7.to_k_ip.weight",
    "7.to_v_ip.weight",
    "9.to_k_ip.weight",
    "9.to_v_ip.weight",
    "11.to_k_ip.weight",
    "11.to_v_ip.weight",
    "13.to_k_ip.weight",
    "13.to_v_ip.weight",
    "15.to_k_ip.weight",
    "15.to_v_ip.weight",
    "17.to_k_ip.weight",
    "17.to_v_ip.weight",
    "19.to_k_ip.weight",
    "19.to_v_ip.weight",
    "21.to_k_ip.weight",
    "21.to_v_ip.weight",
    "23.to_k_ip.weight",
    "23.to_v_ip.weight",
    "25.to_k_ip.weight",
    "25.to_v_ip.weight",
    "27.to_k_ip.weight",
    "27.to_v_ip.weight",
    "29.to_k_ip.weight",
    "29.to_v_ip.weight",
    "31.to_k_ip.weight",
    "31.to_v_ip.weight",
]

# 创建 UNet 到 IP-Adapter 键的映射字典
UNET2IPAadapter_Keys_MAPIING = {
    k: v for k, v in zip(unet_keys_list, ip_adapter_keys_list)
}


def update_unet_ip_adapter_cross_attn_param(
    unet: UNet3DConditionModel, ip_adapter_state_dict: Dict
) -> None:
    """
    使用独立的 IP-Adapter 注意力中的 to_k, to_v 参数更新 UNet
    
    Args:
        unet (UNet3DConditionModel): UNet3D 条件模型
        ip_adapter_state_dict (Dict): IP-Adapter 状态字典，键如 ['1.to_k_ip.weight', '1.to_v_ip.weight', '3.to_k_ip.weight']
    """
    logger.info("开始更新 UNet 的 IP-Adapter 交叉注意力参数")
    
    # 获取 UNet 的空间交叉注意力模块
    unet_spatial_cross_atnns = unet.spatial_cross_attns[0]
    unet_spatial_cross_atnns_dct = {k: v for k, v in unet_spatial_cross_atnns}
    
    # 遍历键映射，将 IP-Adapter 的参数复制到 UNet 中
    for i, (unet_key_more, ip_adapter_key) in enumerate(
        UNET2IPAadapter_Keys_MAPIING.items()
    ):
        ip_adapter_value = ip_adapter_state_dict[ip_adapter_key]
        unet_key_more_spit = unet_key_more.split(".")
        unet_key = ".".join(unet_key_more_spit[:-3])
        suffix = ".".join(unet_key_more_spit[-3:])
        
        logger.debug(
            f"处理第 {i} 个参数: unet_key_more = {unet_key_more}, unet_key = {unet_key}, suffix = {suffix}",
        )
        
        # 根据后缀判断是更新 to_k 还是 to_v 参数
        if "to_k" in suffix:
            with torch.no_grad():
                unet_spatial_cross_atnns_dct[unet_key].to_k_ip.weight.copy_(
                    ip_adapter_value.data
                )
                logger.debug(f"更新 {unet_key}.to_k_ip.weight 参数")
        else:
            with torch.no_grad():
                unet_spatial_cross_atnns_dct[unet_key].to_v_ip.weight.copy_(
                    ip_adapter_value.data
                )
                logger.debug(f"更新 {unet_key}.to_v_ip.weight 参数")
                
    logger.info("完成 UNet 的 IP-Adapter 交叉注意力参数更新")