# Copyright 2023 UC Berkeley Team and The HuggingFace Team. All rights reserved.
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

# 免责声明：此文件深受 https://github.com/ermongroup/ddim 的影响

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
from diffusers.schedulers.scheduling_ddpm import (
    DDPMSchedulerOutput,
    betas_for_alpha_bar,
    DDPMScheduler as DiffusersDDPMScheduler,
)
from ..utils.noise_util import video_fusion_noise

# 设置日志记录器
logger = logging.getLogger(__name__)


class DDPMScheduler(DiffusersDDPMScheduler):
    """
    DDPMScheduler 探索去噪得分匹配和 Langevin 动力学采样之间的联系。

    该模型继承自 SchedulerMixin 和 ConfigMixin。请查看超类文档了解库为所有调度器实现的通用方法，
    例如加载和保存。

    Args:
        num_train_timesteps (`int`, 默认值为 1000):
            训练模型的扩散步数。
        beta_start (`float`, 默认值为 0.0001):
            推理的起始 `beta` 值。
        beta_end (`float`, 默认值为 0.02):
            最终的 `beta` 值。
        beta_schedule (`str`, 默认值为 `"linear"`):
            beta 调度表，是从 beta 范围到用于步进模型的 beta 序列的映射。可选值包括
            `linear`, `scaled_linear`, 或 `squaredcos_cap_v2`。
        variance_type (`str`, 默认值为 `"fixed_small"`):
            在向去噪样本添加噪声时裁剪方差。可选值包括 `fixed_small`, `fixed_small_log`,
            `fixed_large`, `fixed_large_log`, `learned` 或 `learned_range`。
        clip_sample (`bool`, 默认值为 `True`):
            裁剪预测样本以保证数值稳定性。
        clip_sample_range (`float`, 默认值为 1.0):
            样本裁剪的最大幅度。仅在 `clip_sample=True` 时有效。
        prediction_type (`str`, 默认值为 `epsilon`, *可选*):
            调度器函数的预测类型；可以是 `epsilon`（预测扩散过程的噪声）,
            `sample`（直接预测噪声样本）或 `v_prediction`（参见Imagen
            Video 论文第2.4节）。
        thresholding (`bool`, 默认值为 `False`):
            是否使用"动态阈值"方法。这不适用于潜在空间扩散模型，如 Stable Diffusion。
        dynamic_thresholding_ratio (`float`, 默认值为 0.995):
            动态阈值方法的比例。仅在 `thresholding=True` 时有效。
        sample_max_value (`float`, 默认值为 1.0):
            动态阈值的阈值。仅在 `thresholding=True` 时有效。
        timestep_spacing (`str`, 默认值为 `"leading"`):
            时间步的缩放方式。有关详细信息，请参阅 Common Diffusion Noise Schedules and
            Sample Steps are Flawed 的表2。
        steps_offset (`int`, 默认值为 0):
            添加到推理步骤的偏移量。您可以结合使用 `offset=1` 和
            `set_alpha_to_one=False` 使最后一步使用步骤0作为前一个 alpha 乘积，就像在 Stable
            Diffusion 中一样。
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
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1,
        sample_max_value: float = 1,
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
    ):
        """
        初始化 DDPMScheduler
        
        Args:
            num_train_timesteps (int): 训练时间步数
            beta_start (float): 起始 beta 值
            beta_end (float): 结束 beta 值
            beta_schedule (str): beta 调度策略
            trained_betas (ndarray | List[float] | None): 预训练的 betas
            variance_type (str): 方差类型
            clip_sample (bool): 是否裁剪样本
            prediction_type (str): 预测类型
            thresholding (bool): 是否使用阈值处理
            dynamic_thresholding_ratio (float): 动态阈值比例
            clip_sample_range (float): 样本裁剪范围
            sample_max_value (float): 样本最大值
            timestep_spacing (str): 时间步间隔策略
            steps_offset (int): 步数偏移量
        """
        logger.info("初始化 DDPMScheduler")
        logger.debug(f"参数: num_train_timesteps={num_train_timesteps}, beta_start={beta_start}, "
                    f"beta_end={beta_end}, beta_schedule={beta_schedule}, variance_type={variance_type}, "
                    f"clip_sample={clip_sample}, prediction_type={prediction_type}")
        
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            variance_type,
            clip_sample,
            prediction_type,
            thresholding,
            dynamic_thresholding_ratio,
            clip_sample_range,
            sample_max_value,
            timestep_spacing,
            steps_offset,
        )
        logger.info("DDPMScheduler 初始化完成")

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
        w_ind_noise: float = 0.5,
        noise_type: str = "random",
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        通过逆转SDE从上一个时间步预测样本。此函数根据学习到的模型输出（通常为预测噪声）传播扩散过程。

        Args:
            model_output (`torch.FloatTensor`):
                来自学习的扩散模型的直接输出。
            timestep (`float`):
                扩散链中的当前离散时间步。
            sample (`torch.FloatTensor`):
                由扩散过程创建的当前样本实例。
            generator (`torch.Generator`, *可选*):
                随机数生成器。
            return_dict (`bool`, *可选*, 默认值为 `True`):
                是否返回 [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] 或元组。
            w_ind_noise (`float`, *可选*, 默认值为 0.5):
                个体噪声权重。
            noise_type (`str`, *可选*, 默认值为 "random"):
                噪声类型，可以是 "random" 或 "video_fusion"。

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] 或 `tuple`:
                如果 return_dict 为 `True`，则返回 [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`]，
                否则返回元组，其中第一个元素是样本张量。
        """
        logger.info(f"执行 step 操作，timestep={timestep}, noise_type={noise_type}")
        logger.debug(f"输入参数: model_output.shape={model_output.shape}, sample.shape={sample.shape}, "
                    f"w_ind_noise={w_ind_noise}")
        
        t = timestep

        # 获取前一个时间步
        prev_t = self.previous_timestep(t)
        logger.debug(f"当前时间步: {t}, 前一个时间步: {prev_t}")

        # 如果模型输出维度是样本的两倍，并且方差类型为 learned 或 learned_range，则分割输出
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            logger.debug("分割模型输出为预测输出和预测方差")
            model_output, predicted_variance = torch.split(
                model_output, sample.shape[1], dim=1
            )
        else:
            predicted_variance = None
            logger.debug("不使用预测方差")

        # 1. 计算 alphas, betas
        logger.debug("计算 alphas 和 betas")
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        logger.debug(f"alpha_prod_t={alpha_prod_t}, alpha_prod_t_prev={alpha_prod_t_prev}, "
                    f"beta_prod_t={beta_prod_t}, beta_prod_t_prev={beta_prod_t_prev}")

        # 2. 根据预测噪声计算预测的原始样本，也称为公式(15)中的"predicted x_0"
        # 来自 https://arxiv.org/pdf/2006.11239.pdf
        logger.debug(f"使用预测类型: {self.config.prediction_type}")
        if self.config.prediction_type == "epsilon":
            logger.debug("使用 epsilon 预测类型计算原始样本")
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            logger.debug("使用 sample 预测类型")
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            logger.debug("使用 v_prediction 预测类型计算原始样本")
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
        else:
            error_msg = (
                f"prediction_type 给定为 {self.config.prediction_type}，必须是 `epsilon`、`sample` 或"
                " `v_prediction` 中的一个才能用于 DDPMScheduler。"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 3. 裁剪或阈值处理 "predicted x_0"
        if self.config.thresholding:
            logger.debug("应用阈值处理到预测的原始样本")
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            logger.debug(f"裁剪预测的原始样本，范围: [-{self.config.clip_sample_range}, {self.config.clip_sample_range}]")
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. 为 pred_original_sample x_0 和当前样本 x_t 计算系数
        # 参见 https://arxiv.org/pdf/2006.11239.pdf 的公式(7)
        logger.debug("计算预测原始样本系数和当前样本系数")
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        logger.debug(f"pred_original_sample_coeff={pred_original_sample_coeff}, "
                    f"current_sample_coeff={current_sample_coeff}")

        # 5. 计算预测的前一个样本 µ_t
        # 参见 https://arxiv.org/pdf/2006.11239.pdf 的公式(7)
        logger.debug("计算预测的前一个样本")
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )

        # 6. 添加噪声
        variance = 0
        if t > 0:
            logger.debug(f"添加噪声，当前时间步: {t}")
            device = model_output.device
            logger.debug(f"模型输出设备: {device}")

            # 根据噪声类型生成噪声
            if noise_type == "random":
                logger.debug("生成随机噪声")
                variance_noise = randn_tensor(
                    model_output.shape,
                    dtype=model_output.dtype,
                    device=device,
                    generator=generator,
                )
            elif noise_type == "video_fusion":
                logger.debug(f"生成视频融合噪声，w_ind_noise={w_ind_noise}")
                variance_noise = video_fusion_noise(
                    model_output, w_ind_noise=w_ind_noise, generator=generator
                )
            
            # 根据方差类型计算方差
            if self.variance_type == "fixed_small_log":
                logger.debug("使用 fixed_small_log 方差类型")
                variance = (
                    self._get_variance(t, predicted_variance=predicted_variance)
                    * variance_noise
                )
            elif self.variance_type == "learned_range":
                logger.debug("使用 learned_range 方差类型")
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                logger.debug(f"使用默认方差类型: {self.variance_type}")
                variance = (
                    self._get_variance(t, predicted_variance=predicted_variance) ** 0.5
                ) * variance_noise

        # 将方差添加到预测的前一个样本
        pred_prev_sample = pred_prev_sample + variance
        logger.debug(f"最终预测样本 shape: {pred_prev_sample.shape}")

        if not return_dict:
            logger.info("返回元组格式结果")
            return (pred_prev_sample,)

        logger.info("返回 DDPMSchedulerOutput 格式结果")
        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample
        )