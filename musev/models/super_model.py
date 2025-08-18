# -*- coding: utf-8 -*-
from __future__ import annotations

import logging

from typing import Any, Dict, Tuple, Union, Optional
from einops import rearrange, repeat
from torch import nn
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin, load_state_dict

from ..data.data_util import align_repeat_tensor_single_dim

from .unet_3d_condition import UNet3DConditionModel
from .referencenet import ReferenceNet2D
from ip_adapter.ip_adapter import ImageProjModel

logger = logging.getLogger(__name__)


class SuperUNet3DConditionModel(nn.Module):
    """封装了各种子模型的超模型，与 diffusers 的 pipeline 很像，只不过这里是模型定义。
    主要作用
    1. 将支持controlnet、referencenet等功能的计算封装起来，简洁些；
    2. 便于 accelerator 的分布式训练；

    wrap the sub-models, such as unet, referencenet, controlnet, vae, text_encoder, tokenizer, text_emb_extractor, clip_vision_extractor, ip_adapter_image_proj
    1. support controlnet, referencenet, etc.
    2. support accelerator distributed training
    """

    _supports_gradient_checkpointing = True
    print_idx = 0

    # @register_to_config
    def __init__(
        self,
        unet: nn.Module,
        referencenet: nn.Module = None,
        controlnet: nn.Module = None,
        vae: nn.Module = None,
        text_encoder: nn.Module = None,
        tokenizer: nn.Module = None,
        text_emb_extractor: nn.Module = None,
        clip_vision_extractor: nn.Module = None,
        ip_adapter_image_proj: nn.Module = None,
    ) -> None:
        """初始化 SuperUNet3DConditionModel 超模型
        
        Args:
            unet (nn.Module): UNet3D条件模型主体
            referencenet (nn.Module, optional): ReferenceNet2D参考网络模型. Defaults to None.
            controlnet (nn.Module, optional): ControlNet控制网络模型. Defaults to None.
            vae (nn.Module, optional): VAE变分自编码器模型. Defaults to None.
            text_encoder (nn.Module, optional): 文本编码器. Defaults to None.
            tokenizer (nn.Module, optional): 文本分词器. Defaults to None.
            text_emb_extractor (nn.Module, optional): 封装文本编码器和分词器用于文本到嵌入向量的转换. Defaults to None.
            clip_vision_extractor (nn.Module, optional): CLIP视觉特征提取器. Defaults to None.
            ip_adapter_image_proj (nn.Module, optional): IP-Adapter图像投影模型. Defaults to None.
        """
        super().__init__()
        # 存储各个子模型组件
        self.unet = unet
        self.referencenet = referencenet
        self.controlnet = controlnet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_emb_extractor = text_emb_extractor
        self.clip_vision_extractor = clip_vision_extractor
        self.ip_adapter_image_proj = ip_adapter_image_proj

    def forward(
        self,
        unet_params: Dict,
        encoder_hidden_states: torch.Tensor,
        referencenet_params: Dict = None,
        controlnet_params: Dict = None,
        controlnet_scale: float = 1.0,
        vision_clip_emb: Union[torch.Tensor, None] = None,
        prompt_only_use_image_prompt: bool = False,
    ):
        """前向传播函数，处理各种模型组件的协调工作
        
        Args:
            unet_params (Dict): UNet模型参数字典
            encoder_hidden_states (torch.Tensor): 编码器隐藏状态，形状为 b t n d (batch, time, sequence_length, dim)
            referencenet_params (Dict, optional): ReferenceNet模型参数. Defaults to None.
            controlnet_params (Dict, optional): ControlNet模型参数. Defaults to None.
            controlnet_scale (float, optional): ControlNet输出的缩放因子. Defaults to 1.0.
            vision_clip_emb (Union[torch.Tensor, None], optional): 视觉CLIP嵌入向量，形状为 b t d. Defaults to None.
            prompt_only_use_image_prompt (bool, optional): 是否仅使用图像提示. Defaults to False.

        Returns:
            _type_: UNet模型的输出结果
        """
        # 获取批次大小和时间帧数
        batch_size = unet_params["sample"].shape[0]
        time_size = unet_params["sample"].shape[2]
        
        logger.debug(f"开始前向传播，batch_size={batch_size}, time_size={time_size}")

        # ip_adapter_cross_attn, prepare image prompt
        if vision_clip_emb is not None:
            # 处理视觉CLIP嵌入向量，准备图像提示
            if self.print_idx == 0:
                logger.debug(
                    f"vision_clip_emb, before ip_adapter_image_proj, shape={vision_clip_emb.shape} mean={torch.mean(vision_clip_emb)}"
                )
            # 如果是3维张量，增加一个维度
            if vision_clip_emb.ndim == 3:
                vision_clip_emb = rearrange(vision_clip_emb, "b t d-> b t 1 d")
            # 通过IP-Adapter图像投影模型处理
            if self.ip_adapter_image_proj is not None:
                vision_clip_emb = rearrange(vision_clip_emb, "b t n d ->(b t) n d")
                vision_clip_emb = self.ip_adapter_image_proj(vision_clip_emb)
                if self.print_idx == 0:
                    logger.debug(
                        f"vision_clip_emb, after ip_adapter_image_proj shape={vision_clip_emb.shape} mean={torch.mean(vision_clip_emb)}"
                    )
                # 如果是2维张量，增加一个维度
                if vision_clip_emb.ndim == 2:
                    vision_clip_emb = rearrange(vision_clip_emb, "b d-> b 1 d")
                # 重新排列回原来的形状
                vision_clip_emb = rearrange(
                    vision_clip_emb, "(b t) n d -> b t n d", b=batch_size
                )
            # 对时间维度进行对齐和重复
            vision_clip_emb = align_repeat_tensor_single_dim(
                vision_clip_emb, target_length=time_size, dim=1
            )
            if self.print_idx == 0:
                logger.debug(
                    f"vision_clip_emb, after reshape shape={vision_clip_emb.shape} mean={torch.mean(vision_clip_emb)}"
                )

        # 处理编码器隐藏状态和视觉嵌入的相互补充
        if vision_clip_emb is None and encoder_hidden_states is not None:
            vision_clip_emb = encoder_hidden_states
        if vision_clip_emb is not None and encoder_hidden_states is None:
            encoder_hidden_states = vision_clip_emb
        # 当 prompt_only_use_image_prompt 为True时，
        # 1. referencenet 都使用 vision_clip_emb
        # 2. unet 如果没有dual_cross_attn，使用vision_clip_emb，有时不更新
        # 3. controlnet 当前使用 text_prompt

        # when prompt_only_use_image_prompt True,
        # 1. referencenet use vision_clip_emb
        # 2. unet use vision_clip_emb if no dual_cross_attn, sometimes not update
        # 3. controlnet use text_prompt

        # extract referencenet emb
        # 提取ReferenceNet嵌入向量
        if self.referencenet is not None and referencenet_params is not None:
            logger.debug("开始处理ReferenceNet...")
            referencenet_encoder_hidden_states = align_repeat_tensor_single_dim(
                vision_clip_emb,
                target_length=referencenet_params["num_frames"],
                dim=1,
            )
            referencenet_params["encoder_hidden_states"] = rearrange(
                referencenet_encoder_hidden_states, "b t n d->(b t) n d"
            )
            # 执行ReferenceNet前向传播
            referencenet_out = self.referencenet(**referencenet_params)
            (
                down_block_refer_embs,
                mid_block_refer_emb,
                refer_self_attn_emb,
            ) = referencenet_out
            # 记录ReferenceNet输出的详细信息
            if down_block_refer_embs is not None:
                if self.print_idx == 0:
                    logger.debug(
                        f"len(down_block_refer_embs)={len(down_block_refer_embs)}"
                    )
                for i, down_emb in enumerate(down_block_refer_embs):
                    if self.print_idx == 0:
                        logger.debug(
                            f"down_emb, {i}, {down_emb.shape}, mean={down_emb.mean()}"
                        )
            else:
                if self.print_idx == 0:
                    logger.debug(f"down_block_refer_embs is None")
            if mid_block_refer_emb is not None:
                if self.print_idx == 0:
                    logger.debug(
                        f"mid_block_refer_emb, {mid_block_refer_emb.shape}, mean={mid_block_refer_emb.mean()}"
                    )
            else:
                if self.print_idx == 0:
                    logger.debug(f"mid_block_refer_emb is None")
            if refer_self_attn_emb is not None:
                if self.print_idx == 0:
                    logger.debug(f"refer_self_attn_emb, num={len(refer_self_attn_emb)}")
                for i, self_attn_emb in enumerate(refer_self_attn_emb):
                    if self.print_idx == 0:
                        logger.debug(
                            f"referencenet, self_attn_emb, {i}th, shape={self_attn_emb.shape}, mean={self_attn_emb.mean()}"
                        )
            else:
                if self.print_idx == 0:
                    logger.debug(f"refer_self_attn_emb is None")
        else:
            # 如果没有ReferenceNet，设置为None
            down_block_refer_embs, mid_block_refer_emb, refer_self_attn_emb = (
                None,
                None,
                None,
            )

        # extract controlnet emb
        # 提取ControlNet嵌入向量
        if self.controlnet is not None and controlnet_params is not None:
            logger.debug("开始处理ControlNet...")
            controlnet_encoder_hidden_states = align_repeat_tensor_single_dim(
                encoder_hidden_states,
                target_length=unet_params["sample"].shape[2],
                dim=1,
            )
            controlnet_params["encoder_hidden_states"] = rearrange(
                controlnet_encoder_hidden_states, " b t n d -> (b t) n d"
            )
            # 执行ControlNet前向传播
            (
                down_block_additional_residuals,
                mid_block_additional_residual,
            ) = self.controlnet(**controlnet_params)
            # 应用ControlNet缩放因子
            if controlnet_scale != 1.0:
                down_block_additional_residuals = [
                    x * controlnet_scale for x in down_block_additional_residuals
                ]
                mid_block_additional_residual = (
                    mid_block_additional_residual * controlnet_scale
                )
            # 记录ControlNet输出的详细信息
            for i, down_block_additional_residual in enumerate(
                down_block_additional_residuals
            ):
                if self.print_idx == 0:
                    logger.debug(
                        f"{i}, down_block_additional_residual mean={torch.mean(down_block_additional_residual)}"
                    )

            if self.print_idx == 0:
                logger.debug(
                    f"mid_block_additional_residual mean={torch.mean(mid_block_additional_residual)}"
                )
        else:
            # 如果没有ControlNet，设置为None
            down_block_additional_residuals = None
            mid_block_additional_residual = None

        # 根据提示类型决定使用哪种编码器隐藏状态
        if prompt_only_use_image_prompt and vision_clip_emb is not None:
            encoder_hidden_states = vision_clip_emb

        # run unet
        # 执行UNet前向传播
        logger.debug("开始处理UNet...")
        out = self.unet(
            **unet_params,
            down_block_refer_embs=down_block_refer_embs,
            mid_block_refer_emb=mid_block_refer_emb,
            refer_self_attn_emb=refer_self_attn_emb,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            encoder_hidden_states=encoder_hidden_states,
            vision_clip_emb=vision_clip_emb,
        )
        self.print_idx += 1
        logger.debug("前向传播完成")
        return out

    def _set_gradient_checkpointing(self, module, value=False):
        """设置梯度检查点
        
        Args:
            module: 模型模块
            value (bool, optional): 是否启用梯度检查点. Defaults to False.
        """
        if isinstance(module, (UNet3DConditionModel, ReferenceNet2D)):
            module.gradient_checkpointing = value