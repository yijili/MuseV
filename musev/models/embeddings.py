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

from einops import rearrange
import torch
from torch.nn import functional as F
import numpy as np
import logging

from diffusers.models.embeddings import get_2d_sincos_pos_embed_from_grid

# 配置日志记录器
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.info)

# ref diffusers.models.embeddings.get_2d_sincos_pos_embed
def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size_w,
    grid_size_h,
    cls_token=False,
    extra_tokens=0,
    norm_length: bool = False,
    max_length: float = 2048,
):
    """
    创建2D正弦余弦位置嵌入
    
    Args:
        embed_dim (int): 嵌入维度
        grid_size_w (int): 网格宽度
        grid_size_h (int): 网格高度
        cls_token (bool, optional): 是否包含分类token. 默认为 False.
        extra_tokens (int, optional): 额外token数量. 默认为 0.
        norm_length (bool, optional): 是否标准化长度. 默认为 False.
        max_length (float, optional): 最大长度. 默认为 2048.

    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] 或 [1+grid_size*grid_size, embed_dim] (包含或不包含cls_token)
    """
    logger.info(f"开始生成2D正弦余弦位置嵌入，维度: {embed_dim}, 网格尺寸: {grid_size_w}x{grid_size_h}")
    
    # 根据是否标准化长度来决定网格生成方式
    if norm_length and grid_size_h <= max_length and grid_size_w <= max_length:
        # 使用线性空间在0到max_length之间生成网格点
        grid_h = np.linspace(0, max_length, grid_size_h)
        grid_w = np.linspace(0, max_length, grid_size_w)
        logger.debug(f"使用标准化长度生成网格，范围: 0-{max_length}")
    else:
        # 使用默认的整数序列生成网格点
        grid_h = np.arange(grid_size_h, dtype=np.float32)
        grid_w = np.arange(grid_size_w, dtype=np.float32)
        logger.debug(f"使用默认方式生成网格，范围: 0-{grid_size_h-1}, 0-{grid_size_w-1}")
    
    # 创建网格坐标
    grid = np.meshgrid(grid_h, grid_w)  # 这里h优先
    grid = np.stack(grid, axis=0)
    
    # 重塑网格以适应后续处理
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    logger.debug(f"网格重塑为形状: {grid.shape}")
    
    # 调用diffusers库函数生成2D正弦余弦位置嵌入
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    logger.info(f"从网格生成位置嵌入，形状: {pos_embed.shape}")
    
    # 如果需要添加分类token和额外token，则在嵌入前添加零填充
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
        logger.info(f"添加了{extra_tokens}个额外token，新形状: {pos_embed.shape}")
        
    return pos_embed


def resize_spatial_position_emb(
    emb: torch.Tensor,
    height: int,
    width: int,
    scale: float = None,
    target_height: int = None,
    target_width: int = None,
) -> torch.Tensor:
    """
    调整空间位置嵌入的尺寸
    
    Args:
        emb (torch.Tensor): 输入嵌入张量，形状为 b (h w) d
        height (int): 原始高度
        width (int): 原始宽度
        scale (float, optional): 缩放因子. 默认为 None.
        target_height (int, optional): 目标高度. 默认为 None.
        target_width (int, optional): 目标宽度. 默认为 None.

    Returns:
        torch.Tensor: 调整尺寸后的嵌入张量，形状为 b (target_height target_width) d
    """
    logger.info(f"开始调整空间位置嵌入尺寸，原始尺寸: {height}x{width}")
    
    # 如果提供了缩放因子，则根据缩放因子计算目标尺寸
    if scale is not None:
        target_height = int(height * scale)
        target_width = int(width * scale)
        logger.info(f"使用缩放因子{scale}计算目标尺寸: {target_height}x{target_width}")
    else:
        logger.info(f"使用指定目标尺寸: {target_height}x{target_width}")
    
    # 重新排列张量形状以便进行插值
    emb = rearrange(emb, "(h w) (b d) ->b d h w", h=height, b=1)
    logger.debug(f"重新排列嵌入张量，新形状: {emb.shape}")
    
    # 使用双三次插值调整尺寸
    emb = F.interpolate(
        emb,
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
    )
    logger.info(f"使用双三次插值调整尺寸到: {target_height}x{target_width}")
    
    # 重新排列回原始格式
    emb = rearrange(emb, "b d h w-> (h w) (b d)")
    logger.debug(f"重新排列回目标格式，最终形状: {emb.shape}")
    
    return emb