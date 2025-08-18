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

from ip_adapter.resampler import Resampler
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.ip_adapter_faceid import ProjPlusModel, MLPProjModel

from mmcm.vision.feature_extractor.clip_vision_extractor import (
    ImageClipVisionFeatureExtractor,
    ImageClipVisionFeatureExtractorV2,
)
from mmcm.vision.feature_extractor.insight_face_extractor import (
    InsightFaceExtractorNormEmb,
)


from .unet_loader import update_unet_with_sd
from .unet_3d_condition import UNet3DConditionModel
from .ip_adapter_loader import ip_adapter_keys_list

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# refer https://github.com/tencent-ailab/IP-Adapter/issues/168#issuecomment-1846771651
# 定义UNet中与IP-Adapter面部相关的注意力层键列表
unet_keys_list = [
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_k_ip.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.processor.ip_adapter_face_to_v_ip.weight",
]


# 定义UNet键与IP-Adapter键之间的映射关系
UNET2IPAadapter_Keys_MAPIING = {
    k: v for k, v in zip(unet_keys_list, ip_adapter_keys_list)
}


def load_ip_adapter_face_extractor_and_proj_by_name(
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
) -> nn.Module:
    """
    根据模型名称加载IP-Adapter面部特征提取器和投影模型
    
    Args:
        model_name: 模型名称，目前只支持"IPAdapterFaceID"
        ip_ckpt: IP-Adapter检查点路径或模型
        ip_image_encoder: IP图像编码器路径或模型
        cross_attention_dim: 交叉注意力维度
        clip_embeddings_dim: CLIP嵌入维度
        clip_extra_context_tokens: 额外上下文token数量
        ip_scale: IP缩放因子
        dtype: 数据类型
        device: 设备
        unet: UNet模型
        
    Returns:
        元组，包含面部特征提取器和图像投影模型
    """
    logger.info(f"开始加载IP-Adapter面部提取器和投影模型，模型名称: {model_name}")
    
    if model_name == "IPAdapterFaceID":
        # 如果提供了图像编码器，则初始化InsightFace特征提取器
        if ip_image_encoder is not None:
            logger.info("初始化InsightFaceExtractorNormEmb面部特征提取器")
            ip_adapter_face_emb_extractor = InsightFaceExtractorNormEmb(
                pretrained_model_name_or_path=ip_image_encoder,
                dtype=dtype,
                device=device,
            )
        else:
            logger.info("未提供图像编码器，设置面部特征提取器为None")
            ip_adapter_face_emb_extractor = None
            
        # 初始化MLP投影模型
        logger.info("初始化MLPProjModel图像投影模型")
        ip_adapter_image_proj = MLPProjModel(
            cross_attention_dim=cross_attention_dim,
            id_embeddings_dim=clip_embeddings_dim,
            num_tokens=clip_extra_context_tokens,
        ).to(device, dtype=dtype)
    else:
        error_msg = f"不支持的model_name={model_name}，仅支持IPAdapter, IPAdapterPlus, IPAdapterFaceID"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    # 加载IP-Adapter检查点
    logger.info(f"从路径 {ip_ckpt} 加载IP-Adapter状态字典")
    ip_adapter_state_dict = torch.load(
        ip_ckpt,
        map_location="cpu",
    )
    
    # 加载图像投影模型的权重
    logger.info("加载图像投影模型权重")
    ip_adapter_image_proj.load_state_dict(ip_adapter_state_dict["image_proj"])
    
    # 如果提供了UNet且状态字典中包含ip_adapter，则更新UNet参数
    if unet is not None and "ip_adapter" in ip_adapter_state_dict:
        logger.info("更新UNet中的IP-Adapter交叉注意力参数")
        update_unet_ip_adapter_cross_attn_param(
            unet,
            ip_adapter_state_dict["ip_adapter"],
        )
        logger.info(
            f"成功使用 {ip_ckpt} 更新unet.spatial_cross_attn_ip_adapter参数"
        )
        
    logger.info("完成IP-Adapter面部提取器和投影模型的加载")
    return (
        ip_adapter_face_emb_extractor,
        ip_adapter_image_proj,
    )


def update_unet_ip_adapter_cross_attn_param(
    unet: UNet3DConditionModel, ip_adapter_state_dict: Dict
) -> None:
    """
    在UNet中使用独立的IP-Adapter注意力中的to_k, to_v参数
    ip_adapter：类似 ['1.to_k_ip.weight', '1.to_v_ip.weight', '3.to_k_ip.weight']
    
    Args:
        unet (UNet3DConditionModel): UNet模型
        ip_adapter_state_dict (Dict): IP-Adapter状态字典
    """
    logger.info("开始更新UNet中的IP-Adapter交叉注意力参数")
    
    # 获取UNet的空间交叉注意力模块
    unet_spatial_cross_atnns = unet.spatial_cross_attns[0]
    unet_spatial_cross_atnns_dct = {k: v for k, v in unet_spatial_cross_atnns}
    
    # 遍历UNet键与IP-Adapter键的映射关系
    for i, (unet_key_more, ip_adapter_key) in enumerate(
        UNET2IPAadapter_Keys_MAPIING.items()
    ):
        # 获取IP-Adapter中的对应值
        ip_adapter_value = ip_adapter_state_dict[ip_adapter_key]
        unet_key_more_spit = unet_key_more.split(".")
        unet_key = ".".join(unet_key_more_spit[:-3])
        suffix = ".".join(unet_key_more_spit[-3:])
        
        logger.debug(
            f"处理第{i}个参数映射: unet_key_more = {unet_key_more}, unet_key={unet_key}, suffix={suffix}",
        )
        
        # 根据后缀更新对应的权重
        if ".ip_adapter_face_to_k" in suffix:
            with torch.no_grad():
                logger.debug(f"更新 {unet_key} 的ip_adapter_face_to_k_ip.weight参数")
                unet_spatial_cross_atnns_dct[
                    unet_key
                ].ip_adapter_face_to_k_ip.weight.copy_(ip_adapter_value.data)
        else:
            with torch.no_grad():
                logger.debug(f"更新 {unet_key} 的ip_adapter_face_to_v_ip.weight参数")
                unet_spatial_cross_atnns_dct[
                    unet_key
                ].ip_adapter_face_to_v_ip.weight.copy_(ip_adapter_value.data)
                
    logger.info("完成UNet中的IP-Adapter交叉注意力参数更新")