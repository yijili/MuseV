# Copyright 2023 TSAIL Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/LuChengTHU/dpm-solver

import math
from typing import List, Optional, Tuple, Union
import logging

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config

try:
    from diffusers.utils import randn_tensor
except:
    from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
    SchedulerOutput,
)

# 设置日志记录器
logger = logging.getLogger(__name__)

# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    创建一个beta调度，该调度将给定的alpha_t_bar函数离散化，该函数定义了(1-beta)的累积乘积
    从t = [0,1]的时间范围。

    包含一个函数alpha_bar，该函数接受参数t并将其转换为(1-beta)的累积乘积
    到扩散过程的那部分。


    参数:
        num_diffusion_timesteps (`int`): 要生成的betas数量。
        max_beta (`float`): 使用的最大beta值；使用小于1的值来
                     防止奇点。

    返回:
        betas (`np.ndarray`): 调度器用于步进模型输出的betas
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    DPM-Solver (及其改进版本DPM-Solver++) 是一种快速的专用高阶求解器，用于扩散ODEs
    具有收敛阶数保证。根据经验，使用DPM-Solver仅需20步即可生成高质量
    样本，即使仅需10步也能生成相当好的样本。

    有关更多详细信息，请参见原始论文: https://arxiv.org/abs/2206.00927 和 https://arxiv.org/abs/2211.01095

    目前，我们支持噪声预测模型和数据预测模型的多步DPM-Solver。我们
    建议在引导采样时使用`solver_order=2`，在无条件采样时使用`solver_order=3`。

    我们还支持Imagen中的"动态阈值"方法 (https://arxiv.org/abs/2205.11487)。对于像素空间
    扩散模型，您可以同时设置`algorithm_type="dpmsolver++"`和`thresholding=True`来使用动态
    阈值。请注意，阈值方法不适用于潜在空间扩散模型（如
    stable-diffusion）。

    我们还支持DPM-Solver和DPM-Solver++的SDE变体，这是一种用于反向的快速SDE求解器
    扩散SDE。目前我们只支持一阶和二阶求解器。我们建议使用
    二阶`sde-dpmsolver++`。

    [`~ConfigMixin`] 负责存储所有通过调度器的[__init__](file://e:\AI\muse\MuseV-main\musev\schedulers\scheduling_dpmsolver_multistep.py#L150-L234)函数传递的配置属性
    例如`num_train_timesteps`。可以通过`scheduler.config.num_train_timesteps`访问它们。
    [`SchedulerMixin`] 通过 [`SchedulerMixin.save_pretrained`] 和
    [`~SchedulerMixin.from_pretrained`] 函数提供常规的加载和保存功能。

    参数:
        num_train_timesteps (`int`): 用于训练模型的扩散步数。
        beta_start (`float`): 推理的起始`beta`值。
        beta_end (`float`): 最终的`beta`值。
        beta_schedule (`str`):
            beta调度，是从beta范围到用于步进模型的beta序列的映射。可选
            `linear`, `scaled_linear`, 或 `squaredcos_cap_v2`。
        trained_betas (`np.ndarray`, 可选):
            直接将beta数组传递给构造函数以绕过`beta_start`, `beta_end`等的选项。
        solver_order (`int`, 默认 `2`):
            DPM-Solver的阶数；可以是`1`或`2`或`3`。我们建议在引导
            采样时使用`solver_order=2`，在无条件采样时使用`solver_order=3`。
        prediction_type (`str`, 默认 `epsilon`, 可选):
            调度器函数的预测类型，`epsilon`之一（预测扩散的噪声
            过程），`sample`（直接预测噪声样本`）或`v_prediction`（参见第2.4节
            https://imagen.research.google/video/paper.pdf）
        thresholding (`bool`, 默认 `False`):
            是否使用"动态阈值"方法（由Imagen引入，https://arxiv.org/abs/2205.11487）。
            对于像素空间扩散模型，您可以同时设置`algorithm_type=dpmsolver++`和`thresholding=True`来
            使用动态阈值。请注意，阈值方法不适用于潜在空间扩散
            模型（如stable-diffusion）。
        dynamic_thresholding_ratio (`float`, 默认 `0.995`):
            动态阈值方法的比率。默认为`0.995`，与Imagen相同
            （https://arxiv.org/abs/2205.11487）。
        sample_max_value (`float`, 默认 `1.0`):
            动态阈值的阈值。仅在`thresholding=True`和
            `algorithm_type="dpmsolver++`时有效。
        algorithm_type (`str`, 默认 `dpmsolver++`):
            求解器的算法类型。`dpmsolver`或`dpmsolver++`或`sde-dpmsolver`或
            `sde-dpmsolver++`。`dpmsolver`类型实现了https://arxiv.org/abs/2206.00927中的算法，而
            `dpmsolver++`类型实现了https://arxiv.org/abs/2211.01095中的算法。我们建议使用
            `dpmsolver++`或`sde-dpmsolver++`与`solver_order=2`进行引导采样（例如stable-diffusion）。
        solver_type (`str`, 默认 `midpoint`):
            二阶求解器的求解器类型。`midpoint`或`heun`。求解器类型稍有影响
            样本质量，特别是对于少量步骤。我们通过经验发现`midpoint`求解器是
            稍好一些，所以我们建议使用`midpoint`类型。
        lower_order_final (`bool`, 默认 `True`):
            是否在最后几步中使用低阶求解器。仅对< 15个推理步骤有效。我们通过经验
            发现这个技巧可以稳定DPM-Solver的采样，对于步骤< 15，特别是步骤<= 10。
        use_karras_sigmas (`bool`, *可选*, 默认为 `False`):
             此参数控制在采样过程中是否使用Karras sigmas（Karras等（2022）方案）来设置噪声调度中的步长大小。
             如果为True，sigma将根据论文https://arxiv.org/pdf/2206.00364.pdf的方程（5）中定义的噪声水平序列{σi}确定。
        lambda_min_clipped (`float`, 默认 `-inf`):
            数值稳定性所需的lambda(t)最小值的裁剪阈值。这对于
            余弦（squaredcos_cap_v2）噪声调度至关重要。
        variance_type (`str`, *可选*):
            设置为"learned"或"learned_range"用于预测方差的扩散模型。例如，OpenAI的
            guided-diffusion（https://github.com/openai/guided-diffusion）预测模型输出中高斯分布的均值和方差。
            DPM-Solver只需要"mean"输出，因为它是基于
            扩散ODEs。模型的输出是否包含预测的高斯方差。例如，OpenAI的
            guided-diffusion（https://github.com/openai/guided-diffusion）预测模型输出中高斯分布的均值和方差。
            DPM-Solver只需要"mean"输出，因为它是基于
            扩散ODEs。
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
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        use_karras_sigmas: Optional[bool] = True,
        lambda_min_clipped: float = -float("inf"),
        variance_type: Optional[str] = None,
    ):
        """
        初始化DPM-Solver多步调度器
        
        参数:
            num_train_timesteps (`int`): 训练时使用的扩散步骤数
            beta_start (`float`): beta起始值
            beta_end (`float`): beta结束值
            beta_schedule (`str`): beta调度类型
            trained_betas (`Optional[Union[np.ndarray, List[float]]]`): 预训练的betas
            solver_order (`int`): 求解器阶数
            prediction_type (`str`): 预测类型
            thresholding (`bool`): 是否使用阈值处理
            dynamic_thresholding_ratio (`float`): 动态阈值比率
            sample_max_value (`float`): 样本最大值
            algorithm_type (`str`): 算法类型
            solver_type (`str`): 求解器类型
            lower_order_final (`bool`): 是否在最后几步使用低阶求解器
            use_karras_sigmas (`Optional[bool]`): 是否使用Karras sigmas
            lambda_min_clipped (`float`): lambda最小裁剪值
            variance_type (`Optional[str]`): 方差类型
        """
        logger.info(f"初始化DPMSolverMultistepScheduler，训练步数: {num_train_timesteps}")
        
        # 根据配置设置betas
        if trained_betas is not None:
            logger.debug("使用预训练的betas")
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            logger.debug("使用线性beta调度")
            self.betas = torch.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "scaled_linear":
            # 这个调度对潜在扩散模型非常特殊
            logger.debug("使用缩放线性beta调度")
            self.betas = (
                torch.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=torch.float32,
                )
                ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide余弦调度
            logger.debug("使用Glide余弦调度")
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(
                f"{beta_schedule} does is not implemented for {self.__class__}"
            )

        # 计算alphas和累积乘积
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 目前我们只支持VP类型的噪声调度
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)

        # 初始噪声分布的标准差
        self.init_noise_sigma = 1.0

        # DPM-Solver的设置
        if algorithm_type not in [
            "dpmsolver",
            "dpmsolver++",
            "sde-dpmsolver",
            "sde-dpmsolver++",
        ]:
            if algorithm_type == "deis":
                logger.warning("算法类型'deis'已弃用，使用'dpmsolver++'")
                self.register_to_config(algorithm_type="dpmsolver++")
            else:
                raise NotImplementedError(
                    f"{algorithm_type} does is not implemented for {self.__class__}"
                )

        if solver_type not in ["midpoint", "heun"]:
            if solver_type in ["logrho", "bh1", "bh2"]:
                logger.warning(f"求解器类型'{solver_type}'不受支持，使用'midpoint'")
                self.register_to_config(solver_type="midpoint")
            else:
                raise NotImplementedError(
                    f"{solver_type} does is not implemented for {self.__class__}"
                )

        # 可设置的值
        self.num_inference_steps = None
        timesteps = np.linspace(
            0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32
        )[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self.use_karras_sigmas = use_karras_sigmas
        
        logger.info(f"调度器初始化完成，算法类型: {algorithm_type}, 求解器阶数: {solver_order}")

    def set_timesteps(
        self, num_inference_steps: int = None, device: Union[str, torch.device] = None
    ):
        """
        设置用于扩散链的时间步。在推理前运行的支持函数。

        参数:
            num_inference_steps (`int`):
                使用预训练模型生成样本时使用的扩散步数。
            device (`str` or `torch.device`, 可选):
                应该将时间步移动到的设备。如果为`None`，则不移动时间步。
        """
        logger.info(f"设置时间步，推理步数: {num_inference_steps}")
        
        # 为了数值稳定性裁剪所有lambda(t)的最小值。
        # 这对余弦（squaredcos_cap_v2）噪声调度至关重要。
        clipped_idx = torch.searchsorted(
            torch.flip(self.lambda_t, [0]), self.config.lambda_min_clipped
        )
        timesteps = (
            np.linspace(
                0,
                self.config.num_train_timesteps - 1 - clipped_idx,
                num_inference_steps + 1,
            )
            .round()[::-1][:-1]
            .copy()
            .astype(np.int64)
        )

        if self.use_karras_sigmas:
            logger.debug("使用Karras sigmas")
            sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
            log_sigmas = np.log(sigmas)
            sigmas = self._convert_to_karras(
                in_sigmas=sigmas, num_inference_steps=num_inference_steps
            )
            timesteps = np.array(
                [self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]
            ).round()
            timesteps = np.flip(timesteps).copy().astype(np.int64)

        # 当num_inference_steps == num_train_timesteps时，我们最终可能会有
        # 时间步中的重复项。
        _, unique_indices = np.unique(timesteps, return_index=True)
        timesteps = timesteps[np.sort(unique_indices)]

        self.timesteps = torch.from_numpy(timesteps).to(device)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0
        
        logger.debug(f"时间步设置完成，实际步数: {len(timesteps)}")

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        """
        "动态阈值：在每个采样步骤中，我们将s设置为xt0中的某个百分位绝对像素值（在时间步t处对x_0的预测），
        如果s > 1，则将xt0阈值化到范围[-s, s]，然后除以s。动态阈值将饱和像素（接近-1和1的像素）
        向内推，从而主动防止像素在每个步骤中饱和。我们发现动态阈值显著改善了
        照片真实感以及图像-文本对齐，特别是在使用非常大的引导权重时。"

        https://arxiv.org/abs/2205.11487
        """
        logger.debug("应用动态阈值处理")
        
        dtype = sample.dtype
        batch_size, channels, height, width = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = (
                sample.float()
            )  # 为分位数计算升级，以及cpu half不实现clamp

        # 展平样本以沿每个图像进行分位数计算
        sample = sample.reshape(batch_size, channels * height * width)

        abs_sample = sample.abs()  # "某个百分位绝对像素值"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # 当裁剪到min=1时，等效于标准的[-1, 1]裁剪

        s = s.unsqueeze(1)  # (batch_size, 1) 因为clamp将沿dim=0广播
        sample = (
            torch.clamp(sample, -s, s) / s
        )  # "我们将xt0阈值化到范围[-s, s]，然后除以s"

        sample = sample.reshape(batch_size, channels, height, width)
        sample = sample.to(dtype)

        return sample

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t
    def _sigma_to_t(self, sigma, log_sigmas):
        """
        将sigma转换为时间步
        
        参数:
            sigma: sigma值
            log_sigmas: 对数sigmas
            
        返回:
            转换后的时间步
        """
        logger.debug("将sigma转换为时间步")
        
        # 获取对数sigma
        log_sigma = np.log(sigma)

        # 获取分布
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # 获取sigmas范围
        low_idx = (
            np.cumsum((dists >= 0), axis=0)
            .argmax(axis=0)
            .clip(max=log_sigmas.shape[0] - 2)
        )
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # 插值sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # 将插值转换为时间范围
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    def _convert_to_karras(
        self, in_sigmas: torch.FloatTensor, num_inference_steps
    ) -> torch.FloatTensor:
        """构建Karras等（2022）的噪声调度。"""
        logger.debug("转换为Karras噪声调度")

        sigma_min: float = in_sigmas[-1].item()
        sigma_max: float = in_sigmas[0].item()

        rho = 7.0  # 7.0是论文中使用的值
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def convert_model_output(
        self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        将模型输出转换为算法（DPM-Solver / DPM-Solver++）所需的相应类型。

        DPM-Solver设计用于离散化噪声预测模型的积分，而DPM-Solver++设计用于
        离散化数据预测模型的积分。因此，我们需要首先将模型输出转换为
        相应的类型以匹配算法。

        注意算法类型和模型类型是解耦的。也就是说，我们可以对噪声预测模型和数据预测模型都使用DPM-Solver或
        DPM-Solver++。

        参数:
            model_output (`torch.FloatTensor`): 来自学习扩散模型的直接输出。
            timestep (`int`): 扩散链中的当前离散时间步。
            sample (`torch.FloatTensor`):
                扩散过程当前正在创建的样本实例。

        返回:
            `torch.FloatTensor`: 转换后的模型输出。
        """
        logger.debug(f"转换模型输出，时间步: {timestep}")

        # DPM-Solver++需要求解数据预测模型的积分。
        if self.config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.config.prediction_type == "epsilon":
                # DPM-Solver和DPM-Solver++只需要"mean"输出。
                if self.config.variance_type in ["learned", "learned_range"]:
                    model_output = model_output[:, :3]
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                logger.debug("应用阈值处理")
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred

        # DPM-Solver需要求解噪声预测模型的积分。
        elif self.config.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            if self.config.prediction_type == "epsilon":
                # DPM-Solver和DPM-Solver++只需要"mean"输出。
                if self.config.variance_type in ["learned", "learned_range"]:
                    epsilon = model_output[:, :3]
                else:
                    epsilon = model_output
            elif self.config.prediction_type == "sample":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = (sample - alpha_t * model_output) / sigma_t
            elif self.config.prediction_type == "v_prediction":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                logger.debug("应用阈值处理")
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = (sample - sigma_t * epsilon) / alpha_t
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t

            return epsilon

    def dpm_solver_first_order_update(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        一阶DPM-Solver的一次迭代（等效于DDIM）。

        详见 https://arxiv.org/abs/2206.00927 的详细推导。

        参数:
            model_output (`torch.FloatTensor`): 来自学习扩散模型的直接输出。
            timestep (`int`): 扩散链中的当前离散时间步。
            prev_timestep (`int`): 扩散链中的前一个离散时间步。
            sample (`torch.FloatTensor`):
                扩散过程当前正在创建的样本实例。

        返回:
            `torch.FloatTensor`: 前一个时间步的样本张量。
        """
        logger.debug(f"执行一阶DPM-Solver更新，时间步: {timestep} -> {prev_timestep}")
        
        lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
        alpha_t, alpha_s = self.alpha_t[prev_timestep], self.alpha_t[timestep]
        sigma_t, sigma_s = self.sigma_t[prev_timestep], self.sigma_t[timestep]
        h = lambda_t - lambda_s
        
        if self.config.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (
                alpha_t * (torch.exp(-h) - 1.0)
            ) * model_output
        elif self.config.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (
                sigma_t * (torch.exp(h) - 1.0)
            ) * model_output
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                (alpha_t / alpha_s) * sample
                - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            )
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        二阶多步DPM-Solver的一次迭代。

        参数:
            model_output_list (`List[torch.FloatTensor]`):
                当前和后续时间步来自学习扩散模型的直接输出。
            timestep (`int`): 扩散链中的当前和后续离散时间步。
            prev_timestep (`int`): 扩散链中的前一个离散时间步。
            sample (`torch.FloatTensor`):
                扩散过程当前正在创建的样本实例。

        返回:
            `torch.FloatTensor`: 前一个时间步的样本张量。
        """
        logger.debug(f"执行二阶多步DPM-Solver更新，前一个时间步: {prev_timestep}")
        
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = (
            self.lambda_t[t],
            self.lambda_t[s0],
            self.lambda_t[s1],
        )
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        
        if self.config.algorithm_type == "dpmsolver++":
            # 详见 https://arxiv.org/abs/2211.01095 的详细推导
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.config.algorithm_type == "dpmsolver":
            # 详见 https://arxiv.org/abs/2206.00927 的详细推导
            if self.config.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                )
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * (torch.exp(h) - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
        return x_t

    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        三阶多步DPM-Solver的一次迭代。

        参数:
            model_output_list (`List[torch.FloatTensor]`):
                当前和后续时间步来自学习扩散模型的直接输出。
            timestep (`int`): 扩散链中的当前和后续离散时间步。
            prev_timestep (`int`): 扩散链中的前一个离散时间步。
            sample (`torch.FloatTensor`):
                扩散过程当前正在创建的样本实例。

        返回:
            `torch.FloatTensor`: 前一个时间步的样本张量。
        """
        logger.debug(f"执行三阶多步DPM-Solver更新，前一个时间步: {prev_timestep}")
        
        t, s0, s1, s2 = (
            prev_timestep,
            timestep_list[-1],
            timestep_list[-2],
            timestep_list[-3],
        )
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        lambda_t, lambda_s0, lambda_s1, lambda_s2 = (
            self.lambda_t[t],
            self.lambda_t[s0],
            self.lambda_t[s1],
            self.lambda_t[s2],
        )
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        
        if self.config.algorithm_type == "dpmsolver++":
            # 详见 https://arxiv.org/abs/2206.00927 的详细推导
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
            )
        elif self.config.algorithm_type == "dpmsolver":
            # 详见 https://arxiv.org/abs/2206.00927 的详细推导
            x_t = (
                (alpha_t / alpha_s0) * sample
                - (sigma_t * (torch.exp(h) - 1.0)) * D0
                - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                - (sigma_t * ((torch.exp(h) - 1.0 - h) / h**2 - 0.5)) * D2
            )
        return x_t

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
        w_ind_noise: float = 0.5,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        步进函数，使用多步DPM-Solver传播样本。

        参数:
            model_output (`torch.FloatTensor`): 来自学习扩散模型的直接输出。
            timestep (`int`): 扩散链中的当前离散时间步。
            sample (`torch.FloatTensor`):
                扩散过程当前正在创建的样本实例。
            return_dict (`bool`): 返回元组而不是SchedulerOutput类的选项

        返回:
            [`~scheduling_utils.SchedulerOutput`] 或 `tuple`: 如果`return_dict`为
            True则为[`~scheduling_utils.SchedulerOutput`]，否则为`tuple`。返回元组时，第一个元素是样本张量。

        """
        logger.debug(f"执行调度器步进，当前时间步: {timestep}")
        
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = len(self.timesteps) - 1
        else:
            step_index = step_index.item()
        prev_timestep = (
            0
            if step_index == len(self.timesteps) - 1
            else self.timesteps[step_index + 1]
        )
        lower_order_final = (
            (step_index == len(self.timesteps) - 1)
            and self.config.lower_order_final
            and len(self.timesteps) < 15
        )
        lower_order_second = (
            (step_index == len(self.timesteps) - 2)
            and self.config.lower_order_final
            and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, timestep, sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        if self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            logger.debug("生成噪声用于SDE算法")
            #             noise = randn_tensor(
            #                 model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            #             )
            common_noise = torch.randn(
                model_output.shape[:2] + (1,) + model_output.shape[3:],
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )  # 公共噪声
            ind_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            s = torch.tensor(
                w_ind_noise, device=model_output.device, dtype=model_output.dtype
            ).to(device)
            noise = torch.sqrt(1 - s) * common_noise + torch.sqrt(s) * ind_noise

        else:
            noise = None

        if (
            self.config.solver_order == 1
            or self.lower_order_nums < 1
            or lower_order_final
        ):
            logger.debug("使用一阶更新")
            prev_sample = self.dpm_solver_first_order_update(
                model_output, timestep, prev_timestep, sample, noise=noise
            )
        elif (
            self.config.solver_order == 2
            or self.lower_order_nums < 2
            or lower_order_second
        ):
            logger.debug("使用二阶更新")
            timestep_list = [self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, timestep_list, prev_timestep, sample, noise=noise
            )
        else:
            logger.debug("使用三阶更新")
            timestep_list = [
                self.timesteps[step_index - 2],
                self.timesteps[step_index - 1],
                timestep,
            ]
            prev_sample = self.multistep_dpm_solver_third_order_update(
                self.model_outputs, timestep_list, prev_timestep, sample
            )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        if not return_dict:
            logger.debug("返回元组格式结果")
            return (prev_sample,)

        logger.debug("返回SchedulerOutput格式结果")
        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(
        self, sample: torch.FloatTensor, *args, **kwargs
    ) -> torch.FloatTensor:
        """
        确保与需要根据
        当前时间步缩放去噪模型输入的调度器的可互换性。

        参数:
            sample (`torch.FloatTensor`): 输入样本

        返回:
            `torch.FloatTensor`: 缩放后的输入样本
        """
        logger.debug("缩放模型输入")
        return sample

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        向原始样本添加噪声
        
        参数:
            original_samples: 原始样本
            noise: 噪声
            timesteps: 时间步
            
        返回:
            添加噪声后的样本
        """
        logger.debug("向样本添加噪声")
        
        # 确保alphas_cumprod和timestep与original_samples具有相同的设备和数据类型
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps