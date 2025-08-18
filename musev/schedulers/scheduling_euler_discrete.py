# 导入必要的模块和类型
from __future__ import annotations
import logging

from typing import List, Optional, Tuple, Union
import numpy as np
from numpy import ndarray
import torch
from torch import Generator, FloatTensor
from diffusers.schedulers.scheduling_euler_discrete import (
    EulerDiscreteScheduler as DiffusersEulerDiscreteScheduler,
    EulerDiscreteSchedulerOutput,
)
from diffusers.utils.torch_utils import randn_tensor

from ..utils.noise_util import video_fusion_noise

# 创建日志记录器实例
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class EulerDiscreteScheduler(DiffusersEulerDiscreteScheduler):
    """
    扩展的欧拉离散调度器，用于扩散模型的推理过程
    支持视频融合噪声等自定义功能
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: ndarray | List[float] | None = None,
        prediction_type: str = "epsilon",
        interpolation_type: str = "linear",
        use_karras_sigmas: bool | None = False,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
    ):
        """
        初始化欧拉离散调度器
        
        Args:
            num_train_timesteps: 训练时的时间步数
            beta_start: beta参数的起始值
            beta_end: beta参数的结束值
            beta_schedule: beta调度策略
            trained_betas: 预训练的beta值
            prediction_type: 预测类型 ("epsilon", "sample", "v_prediction")
            interpolation_type: 插值类型
            use_karras_sigmas: 是否使用Karras sigmas
            timestep_spacing: 时间步间距策略
            steps_offset: 时间步偏移量
        """
        # 调用父类初始化方法
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            prediction_type,
            interpolation_type,
            use_karras_sigmas,
            timestep_spacing,
            steps_offset,
        )
        # 记录初始化日志
        logger.info("EulerDiscreteScheduler 初始化完成")

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        w_ind_noise: float = 0.5,
        noise_type: str = "random",
    ) -> Union[EulerDiscreteSchedulerOutput, Tuple]:
        """
        通过反转SDE从上一个时间步预测样本。此函数根据学习到的模型输出（通常为预测噪声）传播扩散过程。

        Args:
            model_output (`torch.FloatTensor`):
                来自学习的扩散模型的直接输出。
            timestep (`float`):
                扩散链中的当前离散时间步。
            sample (`torch.FloatTensor`):
                由扩散过程创建的样本的当前实例。
            s_churn (`float`):
                控制随机扰动的参数
            s_tmin  (`float`):
                应用随机扰动的最小sigma值
            s_tmax  (`float`):
                应用随机扰动的最大sigma值
            s_noise (`float`, defaults to 1.0):
                添加到样本中的噪声的缩放因子。
            generator (`torch.Generator`, *optional*):
                随机数生成器。
            return_dict (`bool`):
                是否返回 [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] 或元组。
            w_ind_noise (`float`):
                视频融合噪声的独立噪声权重
            noise_type (`str`):
                噪声类型 ("random" 或 "video_fusion")

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] 或 `tuple`:
                如果 return_dict 为 `True`，则返回 [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`]，
                否则返回一个元组，其中第一个元素是样本张量。
        """

        # 检查时间步是否为整数类型，如果是则抛出异常
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

        # 检查是否调用了scale_model_input函数
        if not self.is_scale_input_called:
            logger.warning(
                "The scale_model_input function should be called before step to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        # 初始化步骤索引
        if self.step_index is None:
            self._init_step_index(timestep)
            logger.debug(f"初始化步骤索引，当前时间步: {timestep}")

        # 获取当前sigma值
        sigma = self.sigmas[self.step_index]
        logger.debug(f"获取当前sigma值: {sigma}")

        # 计算gamma值，用于控制随机扰动
        gamma = (
            min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigma <= s_tmax
            else 0.0
        )
        logger.debug(f"计算gamma值: {gamma}")
        
        device = model_output.device
        logger.debug(f"模型输出设备: {device}")

        # 根据噪声类型生成噪声
        if noise_type == "random":
            logger.debug("生成随机噪声")
            noise = randn_tensor(
                model_output.shape,
                dtype=model_output.dtype,
                device=device,
                generator=generator,
            )
        elif noise_type == "video_fusion":
            logger.debug(f"生成视频融合噪声，独立噪声权重: {w_ind_noise}")
            noise = video_fusion_noise(
                model_output, w_ind_noise=w_ind_noise, generator=generator
            )

        # 计算eps和sigma_hat
        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)
        logger.debug(f"计算eps和sigma_hat: eps.shape={eps.shape}, sigma_hat={sigma_hat}")

        # 如果gamma大于0，则对样本添加噪声扰动
        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5
            logger.debug("应用gamma扰动到样本")

        # 1. 根据sigma缩放的预测噪声计算预测的原始样本(x_0)
        # 注意: "original_sample"不应作为预期的prediction_type，但为了向后兼容性而保留
        if (
            self.config.prediction_type == "original_sample"
            or self.config.prediction_type == "sample"
        ):
            pred_original_sample = model_output
            logger.debug("使用模型输出作为预测的原始样本 (original_sample/sample prediction_type)")
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
            logger.debug("使用epsilon预测类型计算原始样本")
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (
                sample / (sigma**2 + 1)
            )
            logger.debug("使用v_prediction预测类型计算原始样本")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        # 2. 转换为ODE导数
        derivative = (sample - pred_original_sample) / sigma_hat
        logger.debug(f"计算ODE导数，derivative shape: {derivative.shape}")

        # 计算时间步长
        dt = self.sigmas[self.step_index + 1] - sigma_hat
        logger.debug(f"计算时间步长dt: {dt}")

        # 计算前一样本
        prev_sample = sample + derivative * dt
        logger.debug(f"计算前一样本，prev_sample shape: {prev_sample.shape}")

        # 完成后将步骤索引增加1
        self._step_index += 1
        logger.debug("步骤索引增加1")

        # 根据return_dict参数决定返回格式
        if not return_dict:
            logger.debug("返回元组格式结果")
            return (prev_sample,)

        logger.debug("返回EulerDiscreteSchedulerOutput格式结果")
        return EulerDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )

    def step_bk(
        self,
        model_output: FloatTensor,
        timestep: float | FloatTensor,
        sample: FloatTensor,
        s_churn: float = 0,
        s_tmin: float = 0,
        s_tmax: float = float("inf"),
        s_noise: float = 1,
        generator: Generator | None = None,
        return_dict: bool = True,
        w_ind_noise: float = 0.5,
        noise_type: str = "random",
    ) -> EulerDiscreteSchedulerOutput | Tuple:
        """
        通过反转SDE预测前一个时间步的样本。这是从学习到的模型输出（通常为预测噪声）传播扩散过程的核心函数。

        Args:
            model_output (`torch.FloatTensor`): 来自学习的扩散模型的直接输出。
            timestep (`float`): 扩散链中的当前时间步。
            sample (`torch.FloatTensor`):
                正在由扩散过程创建的样本的当前实例。
            s_churn (`float`): 控制随机扰动的参数
            s_tmin  (`float`): 应用随机扰动的最小sigma值
            s_tmax  (`float`): 应用随机扰动的最大sigma值
            s_noise (`float`): 噪声缩放因子
            generator (`torch.Generator`, optional): 随机数生成器。
            return_dict (`bool`): 选择返回元组而不是EulerDiscreteSchedulerOutput类
            w_ind_noise (`float`): 视频融合噪声的独立噪声权重
            noise_type (`str`): 噪声类型 ("random" 或 "video_fusion")

        Returns:
            [`~schedulers.scheduling_utils.EulerDiscreteSchedulerOutput`] 或 `tuple`:
            如果 `return_dict` 为 True，则返回 [`~schedulers.scheduling_utils.EulerDiscreteSchedulerOutput`]，
            否则返回 `tuple`。当返回元组时，第一个元素是样本张量。

        """

        # 检查时间步是否为整数类型
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

        # 检查是否调用了scale_model_input函数
        if not self.is_scale_input_called:
            logger.warning(
                "The scale_model_input function should be called before [step](file://e:\AI\muse\MuseV-main\musev\schedulers\scheduling_euler_discrete.py#L46-L172) to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        # 确保时间步在正确的设备上
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
            logger.debug(f"将时间步移动到设备: {self.timesteps.device}")

        # 查找时间步索引
        step_index = (self.timesteps == timestep).nonzero().item()
        logger.debug(f"查找时间步索引: {step_index}")
        
        # 获取sigma值
        sigma = self.sigmas[step_index]
        logger.debug(f"获取sigma值: {sigma}")

        # 计算gamma值
        gamma = (
            min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigma <= s_tmax
            else 0.0
        )
        logger.debug(f"计算gamma值: {gamma}")

        device = model_output.device
        logger.debug(f"模型输出设备: {device}")
        
        # 根据噪声类型生成噪声
        if noise_type == "random":
            logger.debug("生成随机噪声")
            noise = randn_tensor(
                model_output.shape,
                dtype=model_output.dtype,
                device=device,
                generator=generator,
            )
        elif noise_type == "video_fusion":
            logger.debug(f"生成视频融合噪声，独立噪声权重: {w_ind_noise}")
            noise = video_fusion_noise(
                model_output, w_ind_noise=w_ind_noise, generator=generator
            )
            
        # 计算eps和sigma_hat
        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)
        logger.debug(f"计算eps和sigma_hat: eps.shape={eps.shape}, sigma_hat={sigma_hat}")

        # 如果gamma大于0，则对样本添加噪声扰动
        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5
            logger.debug("应用gamma扰动到样本")

        # 1. 根据sigma缩放的预测噪声计算预测的原始样本(x_0)
        # 注意: "original_sample"不应作为预期的prediction_type，但为了向后兼容性而保留
        if (
            self.config.prediction_type == "original_sample"
            or self.config.prediction_type == "sample"
        ):
            pred_original_sample = model_output
            logger.debug("使用模型输出作为预测的原始样本 (original_sample/sample prediction_type)")
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
            logger.debug("使用epsilon预测类型计算原始样本")
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (
                sample / (sigma**2 + 1)
            )
            logger.debug("使用v_prediction预测类型计算原始样本")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        # 2. 转换为ODE导数
        derivative = (sample - pred_original_sample) / sigma_hat
        logger.debug(f"计算ODE导数，derivative shape: {derivative.shape}")

        # 计算时间步长
        dt = self.sigmas[step_index + 1] - sigma_hat
        logger.debug(f"计算时间步长dt: {dt}")

        # 计算前一样本
        prev_sample = sample + derivative * dt
        logger.debug(f"计算前一样本，prev_sample shape: {prev_sample.shape}")

        # 根据return_dict参数决定返回格式
        if not return_dict:
            logger.debug("返回元组格式结果")
            return (prev_sample,)

        logger.debug("返回EulerDiscreteSchedulerOutput格式结果")
        return EulerDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )