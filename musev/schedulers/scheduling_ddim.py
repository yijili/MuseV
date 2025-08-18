# Copyright 2023 Stanford University Team and The HuggingFace Team. All rights reserved.
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

# 免责声明：此代码深受 https://github.com/pesser/pytorch_diffusion 和 https://github.com/hojonathanho/diffusion 的影响

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import logging

import numpy as np
from numpy import ndarray
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
)
from diffusers.schedulers.scheduling_ddim import (
    DDIMSchedulerOutput,
    rescale_zero_terminal_snr,
    betas_for_alpha_bar,
    DDIMScheduler as DiffusersDDIMScheduler,
)
from ..utils.noise_util import video_fusion_noise

# 设置日志记录器
logger = logging.getLogger(__name__)


class DDIMScheduler(DiffusersDDIMScheduler):
    """
    DDIMScheduler 使用非马尔可夫指导扩展了去噪扩散概率模型（DDPMs）中引入的去噪过程。

    该模型继承自 SchedulerMixin 和 ConfigMixin。请查看超类文档以了解库为所有调度器实现的通用方法，
    如加载和保存。

    Args:
        num_train_timesteps (`int`, 默认值为 1000):
            训练模型的扩散步数。
        beta_start (`float`, 默认值为 0.0001):
            推理的起始 `beta` 值。
        beta_end (`float`, 默认值为 0.02):
            最终的 `beta` 值。
        beta_schedule (`str`, 默认值为 `"linear"`):
            beta 调度表，是从 beta 范围到用于逐步执行模型的 beta 序列的映射。可选值包括
            `linear`, `scaled_linear`, 或 `squaredcos_cap_v2`。
        trained_betas (`np.ndarray`, *可选*):
            直接传递一个 betas 数组给构造函数以绕过 `beta_start` 和 `beta_end`。
        clip_sample (`bool`, 默认值为 `True`):
            为了数值稳定性而裁剪预测样本。
        clip_sample_range (`float`, 默认值为 1.0):
            样本裁剪的最大幅度。仅在 `clip_sample=True` 时有效。
        set_alpha_to_one (`bool`, 默认值为 `True`):
            每个扩散步骤使用该步骤和前一个步骤的 alpha 乘积值。对于最后一步没有前一个 alpha。
            当此选项为 `True` 时，前一个 alpha 乘积固定为 `1`，否则使用步骤 0 的 alpha 值。
        steps_offset (`int`, 默认值为 0):
            添加到推理步骤的偏移量。可以结合使用 `offset=1` 和 `set_alpha_to_one=False`，
            使最后一步使用步骤 0 作为前一个 alpha 乘积，就像在 Stable Diffusion 中一样。
        prediction_type (`str`, 默认值为 `epsilon`, *可选*):
            调度器函数的预测类型；可以是 `epsilon`（预测扩散过程的噪声），
            `sample`（直接预测噪声样本）或 `v_prediction`（参见 Imagen Video 论文第 2.4 节）。
        thresholding (`bool`, 默认值为 `False`):
            是否使用"动态阈值"方法。这对于潜在空间扩散模型（如 Stable Diffusion）不合适。
        dynamic_thresholding_ratio (`float`, 默认值为 0.995):
            动态阈值方法的比例。仅在 `thresholding=True` 时有效。
        sample_max_value (`float`, 默认值为 1.0):
            动态阈值的阈值。仅在 `thresholding=True` 时有效。
        timestep_spacing (`str`, 默认值为 `"leading"`):
            时间步的缩放方式。有关详细信息，请参阅 Common Diffusion Noise Schedules and Sample Steps are Flawed 的表 2。
        rescale_betas_zero_snr (`bool`, 默认值为 `False`):
            是否重新缩放 betas 以具有零终端信噪比。这使模型能够生成非常亮和暗的样本，
            而不是限制在中等亮度的样本。与 --offset_noise 松散相关。
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: ndarray | List[float] | None = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1,
        sample_max_value: float = 1,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
    ):
        """
        初始化 DDIM 调度器
        
        Args:
            num_train_timesteps (int): 训练时间步数
            beta_start (float): beta 起始值
            beta_end (float): beta 结束值
            beta_schedule (str): beta 调度策略
            trained_betas (ndarray | List[float] | None): 预训练的 betas 数组
            clip_sample (bool): 是否裁剪样本
            set_alpha_to_one (bool): 是否将最终 alpha 设置为 1
            steps_offset (int): 步骤偏移量
            prediction_type (str): 预测类型
            thresholding (bool): 是否使用阈值处理
            dynamic_thresholding_ratio (float): 动态阈值比例
            clip_sample_range (float): 样本裁剪范围
            sample_max_value (float): 样本最大值
            timestep_spacing (str): 时间步间距策略
            rescale_betas_zero_snr (bool): 是否重新缩放 betas 为零信噪比
        """
        logger.info("初始化 DDIMScheduler")
        logger.debug(f"参数: num_train_timesteps={num_train_timesteps}, beta_start={beta_start}, beta_end={beta_end}")
        logger.debug(f"参数: beta_schedule={beta_schedule}, clip_sample={clip_sample}, prediction_type={prediction_type}")
        
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            clip_sample,
            set_alpha_to_one,
            steps_offset,
            prediction_type,
            thresholding,
            dynamic_thresholding_ratio,
            clip_sample_range,
            sample_max_value,
            timestep_spacing,
            rescale_betas_zero_snr,
        )
        logger.info("DDIMScheduler 初始化完成")

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        w_ind_noise: float = 0.5,
        noise_type: str = "random",
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        通过逆转 SDE 从上一个时间步预测样本。此函数根据学习的模型输出（通常是预测的噪声）传播扩散过程。

        Args:
            model_output (`torch.FloatTensor`):
                来自学习的扩散模型的直接输出。
            timestep (`float`):
                扩散链中的当前离散时间步。
            sample (`torch.FloatTensor`):
                由扩散过程创建的样本的当前实例。
            eta (`float`):
                扩散步骤中添加噪声的权重。
            use_clipped_model_output (`bool`, 默认值为 `False`):
                如果为 `True`，则从裁剪的预测原始样本计算"校正的" `model_output`。
                这是必要的，因为当 `self.config.clip_sample` 为 `True` 时，预测的原始样本被裁剪到 [-1, 1]。
                如果没有发生裁剪，"校正的" `model_output` 将与输入提供的值一致，`use_clipped_model_output` 不会产生影响。
            generator (`torch.Generator`, *可选*):
                随机数生成器。
            variance_noise (`torch.FloatTensor`):
                通过直接提供方差本身的噪声来替代使用 `generator` 生成噪声。对于 [`CycleDiffusion`] 等方法很有用。
            return_dict (`bool`, *可选*, 默认值为 `True`):
                是否返回 [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] 或元组。
            w_ind_noise (float): 视频融合噪声的权重
            noise_type (str): 噪声类型，可以是 "random" 或 "video_fusion"

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] 或 `tuple`:
                如果 return_dict 为 `True`，则返回 [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`]，否则返回一个元组，
                其中第一个元素是样本张量。
        """
        logger.info(f"执行 DDIMScheduler.step，timestep={timestep}, eta={eta}, noise_type={noise_type}")
        logger.debug(f"输入参数: model_output.shape={model_output.shape}, sample.shape={sample.shape}")
        logger.debug(f"其他参数: use_clipped_model_output={use_clipped_model_output}, w_ind_noise={w_ind_noise}")
        
        if self.num_inference_steps is None:
            error_msg = "推理步数为 'None'，您需要在创建调度器后运行 'set_timesteps'"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 查看 DDIM 论文公式 (12) 和 (16) https://arxiv.org/pdf/2010.02502.pdf
        # 理想情况下，详细阅读 DDIM 论文以深入理解

        # 符号说明 (<变量名> -> <论文中的名称>)
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) 或 x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "指向 x_t 的方向"
        # - pred_prev_sample -> "x_t-1"

        # 1. 获取前一步的值 (=t-1)
        prev_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )
        logger.debug(f"计算前一时间步: prev_timestep={prev_timestep}")

        # 2. 计算 alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        
        logger.debug(f"Alpha 乘积: alpha_prod_t={alpha_prod_t}, alpha_prod_t_prev={alpha_prod_t_prev}")
        logger.debug(f"Beta 乘积: beta_prod_t={beta_prod_t}")

        # 3. 根据预测的噪声计算预测的原始样本，也称为公式 (12) 中的 "predicted x_0"
        # https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            # 预测类型为 epsilon，直接使用模型输出作为噪声预测
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
            logger.debug("使用 epsilon 预测类型计算原始样本")
        elif self.config.prediction_type == "sample":
            # 预测类型为 sample，模型直接输出去噪后的样本
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
            logger.debug("使用 sample 预测类型计算原始样本")
        elif self.config.prediction_type == "v_prediction":
            # 预测类型为 v_prediction，使用 V 形式的预测
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (
                beta_prod_t**0.5
            ) * sample
            logger.debug("使用 v_prediction 预测类型计算原始样本")
        else:
            error_msg = (
                f"给定的 prediction_type {self.config.prediction_type} 必须是 `epsilon`、`sample` 或 `v_prediction` 之一"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 4. 裁剪或阈值处理 "predicted x_0"
        if self.config.thresholding:
            # 使用动态阈值处理
            pred_original_sample = self._threshold_sample(pred_original_sample)
            logger.debug("应用动态阈值处理到预测的原始样本")
        elif self.config.clip_sample:
            # 使用固定范围裁剪
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )
            logger.debug(f"裁剪预测的原始样本到范围 [{-self.config.clip_sample_range}, {self.config.clip_sample_range}]")

        # 5. 计算方差: "sigma_t(η)" -> 参见公式 (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)
        logger.debug(f"计算方差: variance={variance}, std_dev_t={std_dev_t}")

        if use_clipped_model_output:
            # 在 Glide 中，pred_epsilon 总是从裁剪的 x_0 重新推导出来
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
            logger.debug("使用裁剪后的模型输出重新计算 pred_epsilon")

        # 6. 计算公式 (12) 中指向 x_t 的"方向" https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * pred_epsilon
        logger.debug(f"计算指向 x_t 的方向: pred_sample_direction 的规模可能很大")

        # 7. 计算没有公式 (12) 中"随机噪声"的 x_t https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )
        logger.debug(f"计算去噪后的样本: prev_sample.shape={prev_sample.shape}")

        if eta > 0:
            # 当 eta > 0 时，添加噪声
            if variance_noise is not None and generator is not None:
                error_msg = (
                    "不能同时传递 generator 和 variance_noise。请确保 `generator` 或 `variance_noise` 保持为 `None`。"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            device = model_output.device
            logger.debug(f"生成方差噪声，设备: {device}, 噪声类型: {noise_type}")

            if noise_type == "random":
                # 使用随机噪声
                variance_noise = randn_tensor(
                    model_output.shape,
                    dtype=model_output.dtype,
                    device=device,
                    generator=generator,
                )
                logger.debug("使用随机噪声生成器")
            elif noise_type == "video_fusion":
                # 使用视频融合噪声
                variance_noise = video_fusion_noise(
                    model_output, w_ind_noise=w_ind_noise, generator=generator
                )
                logger.debug(f"使用视频融合噪声，权重: {w_ind_noise}")
            
            variance = std_dev_t * variance_noise
            prev_sample = prev_sample + variance
            logger.debug(f"添加方差噪声到样本，eta={eta}")

        if not return_dict:
            logger.info("返回元组格式结果")
            return (prev_sample,)

        logger.info("返回 DDIMSchedulerOutput 格式结果")
        return DDIMSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )