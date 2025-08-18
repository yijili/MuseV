# Copyright 2023 Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You are may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config

from diffusers.utils import BaseOutput, logging

try:
    from diffusers.utils import randn_tensor
except:
    from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
)

from ..utils.noise_util import video_fusion_noise


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->EulerAncestralDiscrete
class EulerAncestralDiscreteSchedulerOutput(BaseOutput):
    """
    调度器步进函数输出的输出类。

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            前一个时间步计算得到的样本 (x_{t-1})。在去噪循环中，`prev_sample` 应该被用作下一个模型输入。
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            基于当前时间步模型输出预测的去噪样本 (x_{0})。`pred_original_sample` 可用于预览进度或进行引导。
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    创建一个 beta 调度，将给定的 alpha_t_bar 函数离散化，该函数定义了 (1-beta) 从 t = [0,1] 的累积乘积。

    包含一个 alpha_bar 函数，它接受参数 t 并将其转换为 (1-beta) 到该扩散过程部分的累积乘积。

    Args:
        num_diffusion_timesteps (`int`): 要生成的 betas 数量。
        max_beta (`float`): 要使用的最大 beta；使用小于 1 的值来防止奇点。

    Returns:
        betas (`np.ndarray`): 调度器用于步进模型输出的 betas
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class EulerAncestralDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    使用欧拉方法步进的祖先采样。基于 Katherine Crowson 的原始 k-diffusion 实现：
    https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L72

    [`~ConfigMixin`] 负责存储所有在调度器的 [__init__](file://e:\AI\muse\MuseV-main\musev\schedulers\scheduling_euler_ancestral_discrete.py#L119-L169) 函数中传递的配置属性，例如 `num_train_timesteps`。
    它们可以通过 `scheduler.config.num_train_timesteps` 访问。
    [`SchedulerMixin`] 通过 [`SchedulerMixin.save_pretrained`] 和 [`~SchedulerMixin.from_pretrained`] 
    函数提供通用的加载和保存功能。

    Args:
        num_train_timesteps (`int`): 用于训练模型的扩散步数。
        beta_start (`float`): 推理的起始 `beta` 值。
        beta_end (`float`): 最终的 `beta` 值。
        beta_schedule (`str`):
            beta 调度，是从 beta 范围到用于步进模型的 beta 序列的映射。可选 `linear` 或 `scaled_linear`。
        trained_betas (`np.ndarray`, optional):
            直接传递 betas 数组给构造函数以绕过 `beta_start`、`beta_end` 等的选项。
        prediction_type (`str`, default `epsilon`, optional):
            调度器函数的预测类型，可以是 `epsilon`（预测扩散过程的噪声）、`sample`（直接预测噪声样本）
            或 `v_prediction`（参见第 2.4 节 https://imagen.research.google/video/paper.pdf）
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
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
    ):
        """
        初始化 EulerAncestralDiscreteScheduler 实例
        
        Args:
            num_train_timesteps (int): 训练时的时间步数
            beta_start (float): beta 起始值
            beta_end (float): beta 结束值
            beta_schedule (str): beta 调度类型
            trained_betas (Optional[Union[np.ndarray, List[float]]]): 预训练的 betas
            prediction_type (str): 预测类型
        """
        logger.info(f"初始化 EulerAncestralDiscreteScheduler，时间步数: {num_train_timesteps}，beta 调度: {beta_schedule}，预测类型: {prediction_type}")
        
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
            logger.debug("使用预训练的 betas")
        elif beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=torch.float32
            )
            logger.debug("使用线性 beta 调度")
        elif beta_schedule == "scaled_linear":
            # 这个调度专门用于潜在扩散模型
            self.betas = (
                torch.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=torch.float32,
                )
                ** 2
            )
            logger.debug("使用缩放线性 beta 调度")
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
            logger.debug("使用 squaredcos_cap_v2 (Glide 余弦) beta 调度")
        else:
            raise NotImplementedError(
                f"{beta_schedule} does is not implemented for {self.__class__}"
            )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)

        # 初始噪声分布的标准差
        self.init_noise_sigma = self.sigmas.max()

        # 可设置的值
        self.num_inference_steps = None
        timesteps = np.linspace(
            0, num_train_timesteps - 1, num_train_timesteps, dtype=float
        )[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.is_scale_input_called = False

    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        通过 `(sigma**2 + 1) ** 0.5` 缩放去噪模型输入以匹配欧拉算法。

        Args:
            sample (`torch.FloatTensor`): 输入样本
            timestep (`float` or `torch.FloatTensor`): 扩散链中的当前时间步

        Returns:
            `torch.FloatTensor`: 缩放后的输入样本
        """
        logger.debug(f"对模型输入进行缩放，时间步: {timestep}")
        
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)
        self.is_scale_input_called = True
        logger.debug(f"输入缩放完成，sigma: {sigma}")
        return sample

    def set_timesteps(
        self, num_inference_steps: int, device: Union[str, torch.device] = None
    ):
        """
        设置用于扩散链的时间步。在推理前运行的辅助函数。

        Args:
            num_inference_steps (`int`):
                使用预训练模型生成样本时使用的扩散步数。
            device (`str` or `torch.device`, optional):
                时间步应移动到的设备。如果为 `None`，则不移动时间步。
        """
        logger.info(f"设置推理时间步数: {num_inference_steps}，设备: {device}")
        
        self.num_inference_steps = num_inference_steps

        timesteps = np.linspace(
            0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=float
        )[::-1].copy()
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=device)
        if str(device).startswith("mps"):
            # mps 不支持 float64
            self.timesteps = torch.from_numpy(timesteps).to(device, dtype=torch.float32)
        else:
            self.timesteps = torch.from_numpy(timesteps).to(device=device)
        logger.debug(f"时间步设置完成，实际时间步数: {len(self.timesteps)}")

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        w_ind_noise: float = 0.5,
        noise_type: str = "random",
    ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
        """
        通过反转 SDE 预测前一个时间步的样本。这是根据学习到的模型输出（通常是最预测的噪声）传播扩散过程的核心函数。

        Args:
            model_output (`torch.FloatTensor`): 来自学习的扩散模型的直接输出。
            timestep (`float`): 扩散链中的当前时间步。
            sample (`torch.FloatTensor`):
                正在通过扩散过程创建的样本的当前实例。
            generator (`torch.Generator`, optional): 随机数生成器。
            return_dict (`bool`): 返回元组而不是 EulerAncestralDiscreteSchedulerOutput 类的选项

        Returns:
            [`~schedulers.scheduling_utils.EulerAncestralDiscreteSchedulerOutput`] 或 `tuple`:
            如果 `return_dict` 为 True，则返回 [`~schedulers.scheduling_utils.EulerAncestralDiscreteSchedulerOutput`]，否则返回 `tuple`。
            当返回元组时，第一个元素是样本张量。
        """
        logger.debug(f"执行调度器步进，时间步: {timestep}，噪声类型: {noise_type}")

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " EulerDiscreteScheduler.step() is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The [scale_model_input](file://e:\AI\muse\MuseV-main\musev\schedulers\scheduling_euler_ancestral_discrete.py#L171-L190) function should be called before [step](file://e:\AI\muse\MuseV-main\musev\schedulers\scheduling_euler_ancestral_discrete.py#L219-L322) to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]

        # 1. 从 sigma 缩放的预测噪声计算预测的原始样本 (x_0)
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
            logger.debug("使用 epsilon 预测类型计算原始样本")
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (
                sample / (sigma**2 + 1)
            )
            logger.debug("使用 v_prediction 预测类型计算原始样本")
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_from = self.sigmas[step_index]
        sigma_to = self.sigmas[step_index + 1]
        sigma_up = (
            sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
        ) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. 转换为 ODE 导数
        derivative = (sample - pred_original_sample) / sigma

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt

        device = model_output.device
        if noise_type == "random":
            noise = randn_tensor(
                model_output.shape,
                dtype=model_output.dtype,
                device=device,
                generator=generator,
            )
            logger.debug("使用随机噪声")
        elif noise_type == "video_fusion":
            noise = video_fusion_noise(
                model_output, w_ind_noise=w_ind_noise, generator=generator
            )
            logger.debug(f"使用视频融合噪声，权重: {w_ind_noise}")

        prev_sample = prev_sample + noise * sigma_up

        if not return_dict:
            logger.debug("返回元组格式结果")
            return (prev_sample,)

        logger.debug("返回 EulerAncestralDiscreteSchedulerOutput 格式结果")
        return EulerAncestralDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        向原始样本添加噪声
        
        Args:
            original_samples: 原始样本
            noise: 噪声
            timesteps: 时间步
            
        Returns:
            添加噪声后的样本
        """
        logger.debug(f"向样本添加噪声，时间步: {timesteps}")
        
        # 确保 sigmas 和 timesteps 与 original_samples 具有相同的设备和数据类型
        sigmas = self.sigmas.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps 不支持 float64
            schedule_timesteps = self.timesteps.to(
                original_samples.device, dtype=torch.float32
            )
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        logger.debug("噪声添加完成")
        return noisy_samples

    def __len__(self):
        """
        返回训练时间步数
        
        Returns:
            int: 训练时间步数
        """
        return self.config.num_train_timesteps