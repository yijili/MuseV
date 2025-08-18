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

from .referencenet import ReferenceNet2D
from .unet_loader import update_unet_with_sd


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def load_referencenet(
    sd_referencenet_model: Tuple[str, nn.Module],
    sd_model: nn.Module = None,
    need_self_attn_block_embs: bool = False,
    need_block_embs: bool = False,
    dtype: torch.dtype = torch.float16,
    cross_attention_dim: int = 768,
    subfolder: str = "unet",
):
    """
    加载ReferenceNet模型。

    Args:
        sd_referencenet_model (Tuple[str, nn.Module] or str): 预训练的ReferenceNet模型或模型路径。
        sd_model (nn.Module, optional): 用于更新ReferenceNet的sd_model。默认为None。
        need_self_attn_block_embs (bool, optional): 是否计算自注意力块嵌入。默认为False。
        need_block_embs (bool, optional): 是否计算块嵌入。默认为False。
        dtype (torch.dtype, optional): 张量的数据类型。默认为torch.float16。
        cross_attention_dim (int, optional): 交叉注意力的维度。默认为768。
        subfolder (str, optional): 模型的子文件夹。默认为"unet"。

    Returns:
        nn.Module: 加载的ReferenceNet模型。
    """
    logger.info(f"开始加载ReferenceNet模型: {sd_referencenet_model}")
    logger.info(f"参数设置 - need_self_attn_block_embs: {need_self_attn_block_embs}, need_block_embs: {need_block_embs}")
    logger.info(f"参数设置 - dtype: {dtype}, cross_attention_dim: {cross_attention_dim}, subfolder: {subfolder}")

    if isinstance(sd_referencenet_model, str):
        logger.info(f"从路径加载预训练模型: {sd_referencenet_model}")
        referencenet = ReferenceNet2D.from_pretrained(
            sd_referencenet_model,
            subfolder=subfolder,
            need_self_attn_block_embs=need_self_attn_block_embs,
            need_block_embs=need_block_embs,
            torch_dtype=dtype,
            cross_attention_dim=cross_attention_dim,
        )
        logger.info("成功从预训练路径加载ReferenceNet模型")
    elif isinstance(sd_referencenet_model, nn.Module):
        logger.info("使用传入的模型实例作为ReferenceNet")
        referencenet = sd_referencencenet_model
        logger.info("成功使用传入模型实例")
    
    if sd_model is not None:
        logger.info("开始使用sd_model更新ReferenceNet")
        referencenet = update_unet_with_sd(referencenet, sd_model)
        logger.info("成功使用sd_model更新ReferenceNet")
    
    logger.info("ReferenceNet模型加载完成")
    return referencenet


def load_referencenet_by_name(
    model_name: str,
    sd_referencenet_model: Tuple[str, nn.Module],
    sd_model: nn.Module = None,
    cross_attention_dim: int = 768,
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """通过模型名字初始化referencenet，载入预训练参数，
        如希望后续通过简单名字就可以使用预训练模型，需要在这里完成定义
        init referencenet with model_name.
        if you want to use pretrained model with simple name, you need to define it here.
    Args:
        model_name (str): 模型名称
        sd_unet_model (Tuple[str, nn.Module]): unet模型
        sd_model (Tuple[str, nn.Module]): sd模型
        cross_attention_dim (int, optional): 交叉注意力维度. 默认为 768.
        dtype (torch.dtype, optional): 数据类型. 默认为 torch.float16.

    Raises:
        ValueError: 不支持的模型名称

    Returns:
        nn.Module: ReferenceNet模型
    """
    logger.info(f"通过模型名称加载ReferenceNet: {model_name}")
    logger.info(f"参数设置 - cross_attention_dim: {cross_attention_dim}, dtype: {dtype}")
    
    if model_name in [
        "musev_referencenet",
    ]:
        logger.info(f"匹配到支持的模型: {model_name}")
        unet = load_referencenet(
            sd_referencenet_model=sd_referencenet_model,
            sd_model=sd_model,
            cross_attention_dim=cross_attention_dim,
            dtype=dtype,
            need_self_attn_block_embs=False,
            need_block_embs=True,
            subfolder="referencenet",
        )
        logger.info(f"成功加载模型 {model_name}")
    else:
        logger.error(f"不支持的模型名称: {model_name}")
        raise ValueError(
            f"unsupport model_name={model_name}, only support ReferenceNet_V0_block13, ReferenceNet_V1_block13, ReferenceNet_V2_block13, ReferenceNet_V0_sefattn16"
        )
    return unet