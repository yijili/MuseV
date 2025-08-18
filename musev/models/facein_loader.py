# 导入必要的库和模块
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

# 导入视觉特征提取器
from mmcm.vision.feature_extractor.clip_vision_extractor import (
    ImageClipVisionFeatureExtractor,
    ImageClipVisionFeatureExtractorV2,
)
from mmcm.vision.feature_extractor.insight_face_extractor import InsightFaceExtractor

# 导入IP适配器相关模块
from ip_adapter.resampler import Resampler
from ip_adapter.ip_adapter import ImageProjModel

# 导入其他模型加载器
from .unet_loader import update_unet_with_sd
from .unet_3d_condition import UNet3DConditionModel
from .ip_adapter_loader import ip_adapter_keys_list

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# 参考 https://github.com/tencent-ailab/IP-Adapter/issues/168#issuecomment-1846771651
# 定义UNet中FaceIn相关的键列表，这些键对应于UNet中用于面部信息注入的注意力层参数
unet_keys_list = [
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.processor.facein_to_k_ip.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.processor.facein_to_v_ip.weight",
]


# 定义UNet键到IP适配器键的映射关系
UNET2IPAadapter_Keys_MAPIING = {
    k: v for k, v in zip(unet_keys_list, ip_adapter_keys_list)
}


def load_facein_extractor_and_proj_by_name(
    model_name: str,
    ip_ckpt: Tuple[str, nn.Module],
    ip_image_encoder: Tuple[str, nn.Module] = None,
    cross_attention_dim: int = 768,
    clip_embeddings_dim: int = 512,
    clip_extra_context_tokens: int = 1,
    ip_scale: float = 0.0,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    unet: nn.Module = None,
) -> nn.Module:
    """
    根据模型名称加载FaceIn提取器和投影模型
    
    Args:
        model_name (str): 模型名称
        ip_ckpt (Tuple[str, nn.Module]): IP适配器检查点路径和模型
        ip_image_encoder (Tuple[str, nn.Module], optional): IP图像编码器. Defaults to None.
        cross_attention_dim (int, optional): 交叉注意力维度. Defaults to 768.
        clip_embeddings_dim (int, optional): CLIP嵌入维度. Defaults to 512.
        clip_extra_context_tokens (int, optional): CLIP额外上下文token数. Defaults to 1.
        ip_scale (float, optional): IP缩放因子. Defaults to 0.0.
        dtype (torch.dtype, optional): 数据类型. Defaults to torch.float16.
        device (str, optional): 设备. Defaults to "cuda".
        unet (nn.Module, optional): UNet模型. Defaults to None.
        
    Returns:
        nn.Module: 返回加载的模型模块
    """
    # 记录函数调用日志
    logger.info(f"开始加载FaceIn提取器和投影模型，模型名称: {model_name}")
    logger.info(f"参数配置: cross_attention_dim={cross_attention_dim}, clip_embeddings_dim={clip_embeddings_dim}")
    logger.info(f"clip_extra_context_tokens={clip_extra_context_tokens}, ip_scale={ip_scale}")
    
    # 函数体为空，待实现
    pass


def update_unet_facein_cross_attn_param(
    unet: UNet3DConditionModel, ip_adapter_state_dict: Dict
) -> None:
    """
    使用独立的ip_adapter注意力中的to_k, to_v参数更新UNet
    
    该函数将IP适配器中的注意力参数（如'1.to_k_ip.weight', '1.to_v_ip.weight', '3.to_k_ip.weight'等）
    映射并更新到UNet模型中对应的面部信息注入层中
    
    Args:
        unet (UNet3DConditionModel): UNet3D条件模型
        ip_adapter_state_dict (Dict): IP适配器状态字典，包含注意力参数
    """
    # 记录函数调用日志
    logger.info("开始更新UNet的FaceIn交叉注意力参数")
    logger.info(f"UNet模型类型: {type(unet)}")
    logger.info(f"IP适配器状态字典大小: {len(ip_adapter_state_dict)}")
    
    # 函数体为空，待实现
    pass