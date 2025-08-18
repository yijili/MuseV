# 导入必要的库和模块
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import warnings
import os

import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
import PIL
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn.init as init
from diffusers.models.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import is_compiled_module

# ControlnetPredictor类：用于ControlNet推理预测，提取ControlNet骨干网络的嵌入表示
class ControlnetPredictor(object):
    def __init__(self, controlnet_model_path: str, *args, **kwargs):
        """Controlnet 推断函数，用于提取 controlnet backbone的emb，避免训练时重复抽取
            Controlnet inference predictor, used to extract the emb of the controlnet backbone to avoid repeated extraction during training
        Args:
            controlnet_model_path (str): controlnet 模型路径. controlnet model path.
        """
        super(ControlnetPredictor, self).__init__(*args, **kwargs)
        # 从预训练模型路径加载ControlNet模型
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model_path,
        )
        print(f"[ControlnetPredictor] 初始化ControlNet模型，模型路径: {controlnet_model_path}")

    def prepare_image(
        self,
        image,  # b c t h w
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        """
        准备图像数据，进行预处理
        """
        print(f"[ControlnetPredictor.prepare_image] 开始准备图像数据")
        
        # 如果未指定高度和宽度，则从图像张量中获取
        if height is None:
            height = image.shape[-2]
        if width is None:
            width = image.shape[-1]
            
        # 确保高度和宽度是vae_scale_factor的倍数
        width, height = (
            x - x % self.control_image_processor.vae_scale_factor
            for x in (width, height)
        )
        print(f"[ControlnetPredictor.prepare_image] 调整后图像尺寸: 高度={height}, 宽度={width}")
        
        # 重新排列图像张量维度，将时间维度展开为批次维度
        image = rearrange(image, "b c t h w-> (b t) c h w")
        # 将图像数据转换为torch张量并归一化到[0,1]范围
        image = torch.from_numpy(image).to(dtype=torch.float32) / 255.0
        print(f"[ControlnetPredictor.prepare_image] 图像张量形状: {image.shape}")

        # 对图像进行双线性插值以匹配目标尺寸
        image = (
            torch.nn.functional.interpolate(
                image,
                size=(height, width),
                mode="bilinear",
            ),
        )
        print(f"[ControlnetPredictor.prepare_image] 图像插值完成")

        # 检查是否需要进行归一化处理
        do_normalize = self.control_image_processor.config.do_normalize
        if image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False

        # 如果需要归一化，则执行归一化操作
        if do_normalize:
            image = self.control_image_processor.normalize(image)
            print(f"[ControlnetPredictor.prepare_image] 图像归一化完成")

        image_batch_size = image.shape[0]
        print(f"[ControlnetPredictor.prepare_image] 图像批次大小: {image_batch_size}")

        # 根据批次大小确定重复次数
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        # 按指定次数重复图像张量
        image = image.repeat_interleave(repeat_by, dim=0)
        print(f"[ControlnetPredictor.prepare_image] 图像重复处理完成，重复次数: {repeat_by}")

        # 将图像移动到指定设备并转换数据类型
        image = image.to(device=device, dtype=dtype)
        print(f"[ControlnetPredictor.prepare_image] 图像已移动到设备: {device}, 数据类型: {dtype}")

        # 如果启用分类器自由引导且不使用猜测模式，则将图像复制一份用于无条件生成
        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)
            print(f"[ControlnetPredictor.prepare_image] 启用分类器自由引导，图像张量已复制")

        print(f"[ControlnetPredictor.prepare_image] 图像准备完成，最终形状: {image.shape}")
        return image

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        device: str,
        dtype: torch.dtype,
        timesteps: List[float],
        i: int,
        scheduler: KarrasDiffusionSchedulers,
        prompt_embeds: torch.Tensor,
        do_classifier_free_guidance: bool = False,
        # 2b co t ho wo
        latent_model_input: torch.Tensor = None,
        # b co t ho wo
        latents: torch.Tensor = None,
        # b c t h w
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        # b c t(1) hi wi
        controlnet_condition_frames: Optional[torch.FloatTensor] = None,
        # b c t ho wo
        controlnet_latents: Union[torch.FloatTensor, np.ndarray] = None,
        # b c t(1) ho wo
        controlnet_condition_latents: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        return_dict: bool = True,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        latent_index: torch.LongTensor = None,
        vision_condition_latent_index: torch.LongTensor = None,
        **kwargs,
    ):
        """
        ControlNet预测器的主调用函数
        """
        print(f"[ControlnetPredictor.__call__] 开始ControlNet推理，当前时间步: {i}")
        
        # 断言检查：image和controlnet_latents不能同时提供
        assert (
            image is None and controlnet_latents is None
        ), "should set one of image and controlnet_latents"
        print(f"[ControlnetPredictor.__call__] 输入验证通过")

        # 获取ControlNet模型实例，处理编译模块的情况
        controlnet = (
            self.controlnet._orig_mod
            if is_compiled_module(self.controlnet)
            else self.controlnet
        )
        print(f"[ControlnetPredictor.__call__] 获取ControlNet模型实例")

        # 对控制引导参数进行格式对齐
        print(f"[ControlnetPredictor.__call__] 对齐控制引导参数格式")
        if not isinstance(control_guidance_start, list) and isinstance(
            control_guidance_end, list
        ):
            control_guidance_start = len(control_guidance_end) * [
                control_guidance_start
            ]
        elif not isinstance(control_guidance_end, list) and isinstance(
            control_guidance_start, list
        ):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(
            control_guidance_end, list
        ):
            mult = (
                len(controlnet.nets)
                if isinstance(controlnet, MultiControlNetModel)
                else 1
            )
            control_guidance_start, control_guidance_end = mult * [
                control_guidance_start
            ], mult * [control_guidance_end]

        # 处理多ControlNet情况下的条件缩放参数
        if isinstance(controlnet, MultiControlNetModel) and isinstance(
            controlnet_conditioning_scale, float
        ):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
                controlnet.nets
            )

        # 获取全局池化条件设置
        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions
        print(f"[ControlnetPredictor.__call__] 控制引导参数对齐完成")

        # 4. Prepare image - 准备图像数据
        print(f"[ControlnetPredictor.__call__] 开始准备图像数据")
        if isinstance(controlnet, ControlNetModel):
            print(f"[ControlnetPredictor.__call__] 处理单个ControlNet模型")
            if (
                controlnet_latents is not None
                and controlnet_condition_latents is not None
            ):
                print(f"[ControlnetPredictor.__call__] 使用控制网络潜在变量")
                # 处理控制网络潜在变量
                if isinstance(controlnet_latents, np.ndarray):
                    controlnet_latents = torch.from_numpy(controlnet_latents)
                if isinstance(controlnet_condition_latents, np.ndarray):
                    controlnet_condition_latents = torch.from_numpy(
                        controlnet_condition_latents
                    )
                # 连接条件潜在变量和控制网络潜在变量
                controlnet_latents = torch.concat(
                    [controlnet_condition_latents, controlnet_latents], dim=2
                )
                print(f"[ControlnetPredictor.__call__] 控制网络潜在变量连接完成，形状: {controlnet_latents.shape}")
                
                # 如果启用分类器自由引导且不使用猜测模式，则复制潜在变量
                if not guess_mode and do_classifier_free_guidance:
                    controlnet_latents = torch.concat([controlnet_latents] * 2, dim=0)
                # 重新排列张量维度
                controlnet_latents = rearrange(
                    controlnet_latents, "b c t h w->(b t) c h w"
                )
                # 将潜在变量移动到指定设备和数据类型
                controlnet_latents = controlnet_latents.to(device=device, dtype=dtype)
                print(f"[ControlnetPredictor.__call__] 控制网络潜在变量处理完成")
            else:
                print(f"[ControlnetPredictor.__call__] 使用图像帧数据")
                # 使用图像帧数据
                if controlnet_condition_frames is not None:
                    if isinstance(controlnet_condition_frames, np.ndarray):
                        image = np.concatenate(
                            [controlnet_condition_frames, image], axis=2
                        )
                # 调用prepare_image方法准备图像
                image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_videos_per_prompt,
                    num_images_per_prompt=num_videos_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = image.shape[-2:]
                print(f"[ControlnetPredictor.__call__] 图像帧数据处理完成")
        elif isinstance(controlnet, MultiControlNetModel):
            print(f"[ControlnetPredictor.__call__] 处理多个ControlNet模型")
            images = []
            # 处理多ControlNet情况
            if controlnet_latents is not None:
                raise NotImplementedError
            else:
                # 遍历每个图像进行处理
                for i, image_ in enumerate(image):
                    if controlnet_condition_frames is not None and isinstance(
                        controlnet_condition_frames, list
                    ):
                        if isinstance(controlnet_condition_frames[i], np.ndarray):
                            image_ = np.concatenate(
                                [controlnet_condition_frames[i], image_], axis=2
                            )
                    # 准备每个图像
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_videos_per_prompt,
                        num_images_per_prompt=num_videos_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    images.append(image_)

                image = images
                height, width = image[0].shape[-2:]
            print(f"[ControlnetPredictor.__call__] 多ControlNet图像处理完成")
        else:
            assert False

        # 7.1 Create tensor stating which controlnets to keep - 创建控制网络保留张量
        print(f"[ControlnetPredictor.__call__] 创建控制网络保留张量")
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(
                keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
            )

        # 获取当前时间步
        t = timesteps[i]
        print(f"[ControlnetPredictor.__call__] 当前时间步: {t}")

        # controlnet(s) inference - 控制网络推理
        print(f"[ControlnetPredictor.__call__] 开始控制网络推理")
        if guess_mode and do_classifier_free_guidance:
            # 仅对条件批次进行ControlNet推理
            print(f"[ControlnetPredictor.__call__] 使用猜测模式和分类器自由引导")
            control_model_input = latents
            control_model_input = scheduler.scale_model_input(control_model_input, t)
            controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
        else:
            print(f"[ControlnetPredictor.__call__] 使用标准模式")
            control_model_input = latent_model_input
            controlnet_prompt_embeds = prompt_embeds
            
        # 计算条件缩放
        if isinstance(controlnet_keep[i], list):
            cond_scale = [
                c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])
            ]
        else:
            cond_scale = controlnet_conditioning_scale * controlnet_keep[i]
        print(f"[ControlnetPredictor.__call__] 条件缩放计算完成: {cond_scale}")
            
        # 重新排列控制模型输入张量
        control_model_input_reshape = rearrange(
            control_model_input, "b c t h w -> (b t) c h w"
        )
        # 重复编码器隐藏状态
        encoder_hidden_states_repeat = repeat(
            controlnet_prompt_embeds,
            "b n q->(b t) n q",
            t=control_model_input.shape[2],
        )
        print(f"[ControlnetPredictor.__call__] 张量重排和状态重复完成")

        # 执行ControlNet模型推理
        print(f"[ControlnetPredictor.__call__] 执行ControlNet模型推理")
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input_reshape,
            t,
            encoder_hidden_states_repeat,
            controlnet_cond=image,
            controlnet_cond_latents=controlnet_latents,
            conditioning_scale=cond_scale,
            guess_mode=guess_mode,
            return_dict=False,
        )
        print(f"[ControlnetPredictor.__call__] ControlNet推理完成")

        return down_block_res_samples, mid_block_res_sample


# InflatedConv3d类：将2D卷积扩展为3D卷积，用于处理视频数据
class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        """
        前向传播函数，将3D张量在时间维度上分解为2D卷积处理
        """
        print(f"[InflatedConv3d.forward] 开始3D卷积前向传播，输入形状: {x.shape}")
        
        # 获取视频长度（时间维度）
        video_length = x.shape[2]
        print(f"[InflatedConv3d.forward] 视频长度: {video_length}")

        # 重新排列张量，将时间维度合并到批次维度
        x = rearrange(x, "b c f h w -> (b f) c h w")
        print(f"[InflatedConv3d.forward] 重新排列后形状: {x.shape}")
        
        # 调用父类的前向传播（2D卷积）
        x = super().forward(x)
        print(f"[InflatedConv3d.forward] 2D卷积处理完成，形状: {x.shape}")
        
        # 恢复时间维度
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)
        print(f"[InflatedConv3d.forward] 恢复时间维度后形状: {x.shape}")

        return x


def zero_module(module):
    # Zero out the parameters of a module and return it.
    # 将模块的参数置零并返回该模块
    print(f"[zero_module] 开始将模块参数置零")
    for p in module.parameters():
        p.detach().zero_()
    print(f"[zero_module] 模块参数置零完成")
    return module


# PoseGuider类：姿态引导器，用于处理姿态条件数据
class PoseGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        """
        初始化姿态引导器
        """
        super().__init__()
        print(f"[PoseGuider.__init__] 初始化姿态引导器")
        print(f"[PoseGuider.__init__] 条件嵌入通道数: {conditioning_embedding_channels}")
        print(f"[PoseGuider.__init__] 条件通道数: {conditioning_channels}")
        print(f"[PoseGuider.__init__] 块输出通道: {block_out_channels}")
        
        # 输入卷积层
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )
        print(f"[PoseGuider.__init__] 输入卷积层初始化完成")

        # 构建卷积块列表
        self.blocks = nn.ModuleList([])
        print(f"[PoseGuider.__init__] 开始构建卷积块")

        # 遍历块输出通道，构建卷积块
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            # 添加第一个卷积层（保持尺寸）
            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            # 添加第二个卷积层（下采样）
            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )
        print(f"[PoseGuider.__init__] 卷积块构建完成，共{len(self.blocks)}个块")

        # 输出卷积层，参数置零
        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )
        print(f"[PoseGuider.__init__] 输出卷积层初始化完成")

    def forward(self, conditioning):
        """
        前向传播函数
        """
        print(f"[PoseGuider.forward] 开始姿态引导器前向传播，输入形状: {conditioning.shape}")
        
        # 输入卷积处理
        embedding = self.conv_in(conditioning)
        print(f"[PoseGuider.forward] 输入卷积处理完成，形状: {embedding.shape}")
        
        # 应用SiLU激活函数
        embedding = F.silu(embedding)
        print(f"[PoseGuider.forward] SiLU激活函数应用完成")

        # 遍历所有卷积块进行处理
        for i, block in enumerate(self.blocks):
            embedding = block(embedding)
            embedding = F.silu(embedding)
            print(f"[PoseGuider.forward] 第{i+1}个卷积块处理完成，形状: {embedding.shape}")

        # 输出卷积处理
        embedding = self.conv_out(embedding)
        print(f"[PoseGuider.forward] 输出卷积处理完成，最终形状: {embedding.shape}")

        return embedding

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        """
        从预训练模型加载姿态引导器
        """
        print(f"[PoseGuider.from_pretrained] 开始从预训练模型加载姿态引导器")
        print(f"[PoseGuider.from_pretrained] 预训练模型路径: {pretrained_model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(pretrained_model_path):
            print(f"There is no model file in {pretrained_model_path}")
            
        print(
            f"loaded PoseGuider's pretrained weights from {pretrained_model_path} ..."
        )

        # 加载模型状态字典
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        print(f"[PoseGuider.from_pretrained] 模型状态字典加载完成")
        
        # 创建模型实例
        model = PoseGuider(
            conditioning_embedding_channels=conditioning_embedding_channels,
            conditioning_channels=conditioning_channels,
            block_out_channels=block_out_channels,
        )
        print(f"[PoseGuider.from_pretrained] 模型实例创建完成")

        # 加载模型权重
        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"[PoseGuider.from_pretrained] 模型权重加载完成")
        # print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        
        # 计算模型参数量
        params = [p.numel() for n, p in model.named_parameters()]
        print(f"### PoseGuider's Parameters: {sum(params) / 1e6} M")

        return model