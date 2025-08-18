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

# 免责声明：此代码受到 https://github.com/pesser/pytorch_diffusion 和 https://github.com/hojonathanho/diffusion 的强烈影响

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from numpy import ndarray

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers.scheduling_lcm import (
    LCMSchedulerOutput,
    betas_for_alpha_bar,
    rescale_zero_terminal_snr,
    LCMScheduler as DiffusersLCMScheduler,
)
from ..utils.noise_util import video_fusion_noise

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LCMScheduler(DiffusersLCMScheduler):
    """
    扩散模型的LCM（Latent Consistency Models）调度器，用于加速采样过程。
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        trained_betas: ndarray | List[float] | None = None,
        original_inference_steps: int = 50,
        clip_sample: bool = False,
        clip_sample_range: float = 1,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1,
        timestep_spacing: str = "leading",
        timestep_scaling: float = 10,
        rescale_betas_zero_snr: bool = False,
    ):
        """
        初始化LCM调度器
        
        Args:
            num_train_timesteps: 训练时的时间步数
            beta_start: beta分布的起始值
            beta_end: beta分布的结束值
            beta_schedule: beta调度策略
            trained_betas: 预训练的betas值
            original_inference_steps: 原始推理步数
            clip_sample: 是否裁剪样本
            clip_sample_range: 裁剪样本范围
            set_alpha_to_one: 是否将alpha设置为1
            steps_offset: 步数偏移
            prediction_type: 预测类型 ("epsilon", "sample", "v_prediction")
            thresholding: 是否使用阈值处理
            dynamic_thresholding_ratio: 动态阈值比例
            sample_max_value: 样本最大值
            timestep_spacing: 时间步间隔策略
            timestep_scaling: 时间步缩放因子
            rescale_betas_zero_snr: 是否重新缩放betas以实现零SNR
        """
        logger.info("初始化LCMScheduler")
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            original_inference_steps,
            clip_sample,
            clip_sample_range,
            set_alpha_to_one,
            steps_offset,
            prediction_type,
            thresholding,
            dynamic_thresholding_ratio,
            sample_max_value,
            timestep_spacing,
            timestep_scaling,
            rescale_betas_zero_snr,
        )
        logger.debug(f"LCMScheduler初始化完成，参数: num_train_timesteps={num_train_timesteps}, "
                    f"beta_start={beta_start}, beta_end={beta_end}, beta_schedule={beta_schedule}")

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        w_ind_noise: float = 0.5,
        noise_type: str = "random",
    ) -> Union[LCMSchedulerOutput, Tuple]:
        """
        通过反转SDE从上一个时间步预测样本。此函数根据学习到的模型输出（通常是预测的噪声）传播扩散过程。

        Args:
            model_output (`torch.FloatTensor`):
                来自学习扩散模型的直接输出。
            timestep (`float`):
                扩散链中的当前离散时间步。
            sample (`torch.FloatTensor`):
                由扩散过程创建的当前样本实例。
            generator (`torch.Generator`, *optional*):
                随机数生成器。
            return_dict (`bool`, *optional*, defaults to `True`):
                是否返回 [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] 或元组。
            w_ind_noise (`float`, *optional*, defaults to 0.5):
                视频融合噪声的独立噪声权重。
            noise_type (`str`, *optional*, defaults to "random"):
                噪声类型，可以是"random"或"video_fusion"。
                
        Returns:
            [`~schedulers.scheduling_utils.LCMSchedulerOutput`] 或 `tuple`:
                如果 return_dict 为 `True`，则返回 [`~schedulers.scheduling_lcm.LCMSchedulerOutput`]，否则返回元组，
                其中第一个元素是样本张量。
        """
        logger.debug(f"执行step操作，timestep={timestep}, noise_type={noise_type}, w_ind_noise={w_ind_noise}")
        
        # 检查是否已设置推理步数
        if self.num_inference_steps is None:
            logger.error("推理步数为None，需要在创建调度器后运行set_timesteps")
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 初始化步数索引
        if self.step_index is None:
            logger.debug(f"初始化步数索引，timestep={timestep}")
            self._init_step_index(timestep)

        # 1. 获取前一步的值
        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep
        logger.debug(f"当前步索引: {self.step_index}, 前一步索引: {prev_step_index}, 当前时间步: {timestep}, 前一时间步: {prev_timestep}")

        # 2. 计算 alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        logger.debug(f"计算alpha和beta值: alpha_prod_t={alpha_prod_t}, alpha_prod_t_prev={alpha_prod_t_prev}, "
                    f"beta_prod_t={beta_prod_t}, beta_prod_t_prev={beta_prod_t_prev}")

        # 3. 获取边界条件的缩放因子
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)
        logger.debug(f"获取边界条件缩放因子: c_skip={c_skip}, c_out={c_out}")

        # 4. 根据模型参数化计算预测的原始样本 x_0
        if self.config.prediction_type == "epsilon":  # 噪声预测
            predicted_original_sample = (
                sample - beta_prod_t.sqrt() * model_output
            ) / alpha_prod_t.sqrt()
            logger.debug("使用epsilon预测类型计算predicted_original_sample")
        elif self.config.prediction_type == "sample":  # x预测
            predicted_original_sample = model_output
            logger.debug("使用sample预测类型计算predicted_original_sample")
        elif self.config.prediction_type == "v_prediction":  # v预测
            predicted_original_sample = (
                alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
            )
            logger.debug("使用v_prediction预测类型计算predicted_original_sample")
        else:
            logger.error(f"不支持的prediction_type: {self.config.prediction_type}")
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction` for [LCMScheduler](file://e:\AI\muse\MuseV-main\musev\schedulers\scheduling_lcm.py#L41-L311)."
            )
        logger.debug(f"预测的原始样本值范围: [{predicted_original_sample.min()}, {predicted_original_sample.max()}]")

        # 5. 裁剪或阈值处理 "predicted x_0"
        if self.config.thresholding:
            logger.debug("应用阈值处理到predicted_original_sample")
            predicted_original_sample = self._threshold_sample(
                predicted_original_sample
            )
        elif self.config.clip_sample:
            logger.debug(f"裁剪predicted_original_sample到范围 [{-self.config.clip_sample_range}, {self.config.clip_sample_range}]")
            predicted_original_sample = predicted_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 6. 使用边界条件去噪模型输出
        denoised = c_out * predicted_original_sample + c_skip * sample
        logger.debug(f"去噪后样本值范围: [{denoised.min()}, {denoised.max()}]")

        # 7. 为多步推理采样并注入噪声 z ~ N(0, I)
        # 在时间步调度的最后一步不使用噪声。
        # 这也意味着在一步采样中不使用噪声。
        device = model_output.device

        if self.step_index != self.num_inference_steps - 1:
            logger.debug(f"当前不是最后一步，添加噪声。step_index={self.step_index}, num_inference_steps={self.num_inference_steps}")
            if noise_type == "random":
                logger.debug("使用随机噪声")
                noise = randn_tensor(
                    model_output.shape,
                    dtype=model_output.dtype,
                    device=device,
                    generator=generator,
                )
            elif noise_type == "video_fusion":
                logger.debug(f"使用视频融合噪声，w_ind_noise={w_ind_noise}")
                noise = video_fusion_noise(
                    model_output, w_ind_noise=w_ind_noise, generator=generator
                )
            prev_sample = (
                alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
            )
            logger.debug(f"添加噪声后的前一样本值范围: [{prev_sample.min()}, {prev_sample.max()}]")
        else:
            logger.debug("当前是最后一步，不添加噪声")
            prev_sample = denoised

        # 完成后将步数索引增加1
        self._step_index += 1
        logger.debug(f"步数索引增加至: {self._step_index}")

        if not return_dict:
            logger.debug("返回元组格式结果")
            return (prev_sample, denoised)

        logger.debug("返回LCMSchedulerOutput格式结果")
        return LCMSchedulerOutput(prev_sample=prev_sample, denoised=denoised)

    def step_bk(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[LCMSchedulerOutput, Tuple]:
        """
        通过反转SDE从上一个时间步预测样本。此函数根据学习到的模型输出（通常是预测的噪声）传播扩散过程。
        这是原始step方法的备份版本。

        Args:
            model_output (`torch.FloatTensor`):
                来自学习扩散模型的直接输出。
            timestep (`float`):
                扩散链中的当前离散时间步。
            sample (`torch.FloatTensor`):
                由扩散过程创建的当前样本实例。
            generator (`torch.Generator`, *optional*):
                随机数生成器。
            return_dict (`bool`, *optional*, defaults to `True`):
                是否返回 [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] 或元组。
                
        Returns:
            [`~schedulers.scheduling_utils.LCMSchedulerOutput`] 或 `tuple`:
                如果 return_dict 为 `True`，则返回 [`~schedulers.scheduling_lcm.LCMSchedulerOutput`]，否则返回元组，
                其中第一个元素是样本张量。
        """
        logger.debug(f"执行step_bk操作，timestep={timestep}")
        
        # 检查是否已设置推理步数
        if self.num_inference_steps is None:
            logger.error("推理步数为None，需要在创建调度器后运行set_timesteps")
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 初始化步数索引
        if self.step_index is None:
            logger.debug(f"初始化步数索引，timestep={timestep}")
            self._init_step_index(timestep)

        # 1. 获取前一步的值
        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep
        logger.debug(f"当前步索引: {self.step_index}, 前一步索引: {prev_step_index}, 当前时间步: {timestep}, 前一时间步: {prev_timestep}")

        # 2. 计算 alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        logger.debug(f"计算alpha和beta值: alpha_prod_t={alpha_prod_t}, alpha_prod_t_prev={alpha_prod_t_prev}, "
                    f"beta_prod_t={beta_prod_t}, beta_prod_t_prev={beta_prod_t_prev}")

        # 3. 获取边界条件的缩放因子
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)
        logger.debug(f"获取边界条件缩放因子: c_skip={c_skip}, c_out={c_out}")

        # 4. 根据模型参数化计算预测的原始样本 x_0
        if self.config.prediction_type == "epsilon":  # 噪声预测
            predicted_original_sample = (
                sample - beta_prod_t.sqrt() * model_output
            ) / alpha_prod_t.sqrt()
            logger.debug("使用epsilon预测类型计算predicted_original_sample")
        elif self.config.prediction_type == "sample":  # x预测
            predicted_original_sample = model_output
            logger.debug("使用sample预测类型计算predicted_original_sample")
        elif self.config.prediction_type == "v_prediction":  # v预测
            predicted_original_sample = (
                alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
            )
            logger.debug("使用v_prediction预测类型计算predicted_original_sample")
        else:
            logger.error(f"不支持的prediction_type: {self.config.prediction_type}")
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction` for [LCMScheduler](file://e:\AI\muse\MuseV-main\musev\schedulers\scheduling_lcm.py#L41-L311)."
            )
        logger.debug(f"预测的原始样本值范围: [{predicted_original_sample.min()}, {predicted_original_sample.max()}]")

        # 5. 裁剪或阈值处理 "predicted x_0"
        if self.config.thresholding:
            logger.debug("应用阈值处理到predicted_original_sample")
            predicted_original_sample = self._threshold_sample(
                predicted_original_sample
            )
        elif self.config.clip_sample:
            logger.debug(f"裁剪predicted_original_sample到范围 [{-self.config.clip_sample_range}, {self.config.clip_sample_range}]")
            predicted_original_sample = predicted_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 6. 使用边界条件去噪模型输出
        denoised = c_out * predicted_original_sample + c_skip * sample
        logger.debug(f"去噪后样本值范围: [{denoised.min()}, {denoised.max()}]")

        # 7. 为多步推理采样并注入噪声 z ~ N(0, I)
        # 在时间步调度的最后一步不使用噪声。
        # 这也意味着在一步采样中不使用噪声。
        if self.step_index != self.num_inference_steps - 1:
            logger.debug(f"当前不是最后一步，添加噪声。step_index={self.step_index}, num_inference_steps={self.num_inference_steps}")
            noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=denoised.dtype,
            )
            prev_sample = (
                alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
            )
            logger.debug(f"添加噪声后的前一样本值范围: [{prev_sample.min()}, {prev_sample.max()}]")
        else:
            logger.debug("当前是最后一步，不添加噪声")
            prev_sample = denoised

        # 完成后将步数索引增加1
        self._step_index += 1
        logger.debug(f"步数索引增加至: {self._step_index}")

        if not return_dict:
            logger.debug("返回元组格式结果")
            return (prev_sample, denoised)

        logger.debug("返回LCMSchedulerOutput格式结果")
        return LCMSchedulerOutput(prev_sample=prev_sample, denoised=denoised)