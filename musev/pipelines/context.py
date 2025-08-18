# TODO: Adapted from cli
import math
from typing import Callable, List, Optional
import logging

import numpy as np

from mmcm.utils.itertools_util import generate_sample_idxs

# copy from https://github.com/MooreThreads/Moore-AnimateAnyone/blob/master/src/pipelines/context.py

# 配置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ordered_halving(val):
    """
    对输入值进行有序二分操作
    
    Args:
        val: 输入整数值
        
    Returns:
        float: 经过二进制翻转和归一化后的浮点数
    """
    logger.debug(f"执行ordered_halving操作，输入值: {val}")
    bin_str = f"{val:064b}"  # 将整数转换为64位二进制字符串
    bin_flip = bin_str[::-1]  # 翻转二进制字符串
    as_int = int(bin_flip, 2)  # 将翻转后的二进制字符串转换回整数

    result = as_int / (1 << 64)  # 归一化到[0,1)区间
    logger.debug(f"ordered_halving结果: {result}")
    return result


# TODO: closed_loop not work, to fix it
def uniform(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    """
    均匀采样策略，生成上下文帧索引
    
    Args:
        step: 当前步骤索引
        num_steps: 总步骤数
        num_frames: 总帧数
        context_size: 上下文窗口大小
        context_stride: 上下文步长
        context_overlap: 上下文重叠数
        closed_loop: 是否闭合循环
        
    Yields:
        List[int]: 一组上下文帧索引
    """
    logger.info(f"执行uniform采样策略，step={step}, num_frames={num_frames}, context_size={context_size}")
    
    # 如果总帧数小于等于上下文大小，直接返回所有帧索引
    if num_frames <= context_size:
        result = list(range(num_frames))
        logger.debug(f"帧数小于等于上下文大小，直接返回所有帧索引: {result}")
        yield result
        return

    # 限制context_stride不超过计算值
    context_stride = min(
        context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1
    )
    logger.debug(f"调整后的context_stride: {context_stride}")

    # 对每个context_step进行迭代
    for context_step in 1 << np.arange(context_stride):
        logger.debug(f"处理context_step: {context_step}")
        pad = int(round(num_frames * ordered_halving(step)))  # 计算填充值
        logger.debug(f"计算得到pad值: {pad}")
        
        # 生成上下文窗口
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            # 生成具体的帧索引列表
            context = [
                e % num_frames
                for e in range(j, j + context_size * context_step, context_step)
            ]
            logger.debug(f"生成上下文窗口: {context}")
            yield context


def uniform_v2(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    """
    均匀采样策略v2版本，使用generate_sample_idxs工具函数
    
    Args:
        step: 当前步骤索引
        num_steps: 总步骤数
        num_frames: 总帧数
        context_size: 上下文窗口大小
        context_stride: 上下文步长
        context_overlap: 上下文重叠数
        closed_loop: 是否闭合循环
        
    Returns:
        List[int]: 生成的样本索引列表
    """
    logger.info(f"执行uniform_v2采样策略，num_frames={num_frames}, window_size={context_size}")
    result = generate_sample_idxs(
        total=num_frames,
        window_size=context_size,
        step=context_size - context_overlap,
        sample_rate=1,
        drop_last=False,
    )
    logger.debug(f"uniform_v2生成结果: {result}")
    return result


def get_context_scheduler(name: str) -> Callable:
    """
    根据名称获取上下文调度器函数
    
    Args:
        name: 调度器名称 ("uniform" 或 "uniform_v2")
        
    Returns:
        Callable: 对应的调度器函数
        
    Raises:
        ValueError: 当名称不匹配任何已知调度器时抛出异常
    """
    logger.info(f"获取上下文调度器: {name}")
    if name == "uniform":
        logger.debug("返回uniform调度器")
        return uniform
    elif name == "uniform_v2":
        logger.debug("返回uniform_v2调度器")
        return uniform_v2
    else:
        error_msg = f"未知的上下文调度策略: {name}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def get_total_steps(
    scheduler,
    timesteps: List[int],
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    """
    计算总步骤数
    
    Args:
        scheduler: 调度器函数
        timesteps: 时间步列表
        num_steps: 总步骤数
        num_frames: 总帧数
        context_size: 上下文窗口大小
        context_stride: 上下文步长
        context_overlap: 上下文重叠数
        closed_loop: 是否闭合循环
        
    Returns:
        int: 总步骤数
    """
    logger.info(f"计算总步骤数，timesteps长度: {len(timesteps)}")
    total = sum(
        len(
            list(
                scheduler(
                    i,
                    num_steps,
                    num_frames,
                    context_size,
                    context_stride,
                    context_overlap,
                )
            )
        )
        for i in range(len(timesteps))
    )
    logger.debug(f"计算得到总步骤数: {total}")
    return total


def drop_last_repeat_context(contexts: List[List[int]]) -> List[List[int]]:
    """
    如果contexts长度>=2且倒数第二个列表的最后一个元素与最后一个列表的最后一个元素相同，
    则删除最后一个列表（因为它是由于step产生的冗余项）
    
    Args:
        contexts: 上下文列表的列表
        
    Returns:
        List[List[int]]: 处理后的上下文列表
    """
    logger.info(f"处理重复上下文，原始contexts数量: {len(contexts)}")
    if len(contexts) >= 2 and contexts[-1][-1] == contexts[-2][-1]:
        result = contexts[:-1]
        logger.debug("检测到重复上下文，已移除最后一个")
    else:
        result = contexts
        logger.debug("未检测到重复上下文，保持原样")
    logger.debug(f"处理后contexts数量: {len(result)}")
    return result


def prepare_global_context(
    context_schedule: str,
    num_inference_steps: int,
    time_size: int,
    context_frames: int,
    context_stride: int,
    context_overlap: int,
    context_batch_size: int,
):
    """
    准备全局上下文
    
    Args:
        context_schedule: 上下文调度策略名称
        num_inference_steps: 推理步骤数
        time_size: 时间维度大小
        context_frames: 上下文帧数
        context_stride: 上下文步长
        context_overlap: 上下文重叠数
        context_batch_size: 上下文批次大小
        
    Returns:
        List[List[List[int]]]: 全局上下文，按批次组织
    """
    logger.info("开始准备全局上下文")
    logger.debug(f"参数: context_schedule={context_schedule}, num_inference_steps={num_inference_steps}, "
                f"time_size={time_size}, context_frames={context_frames}, context_stride={context_stride}, "
                f"context_overlap={context_overlap}, context_batch_size={context_batch_size}")
    
    # 获取上下文调度器
    context_scheduler = get_context_scheduler(context_schedule)
    logger.debug("已获取上下文调度器")
    
    # 生成上下文队列
    context_queue = list(
        context_scheduler(
            step=0,
            num_steps=num_inference_steps,
            num_frames=time_size,
            context_size=context_frames,
            context_stride=context_stride,
            context_overlap=context_overlap,
        )
    )
    logger.debug(f"生成上下文队列，数量: {len(context_queue)}")
    
    # 如果context_queue的最后一个索引最大值和倒数第二个索引最大值相同，
    # 说明最后一个列表就是因为step带来的冗余项，可以去掉
    # remove the last context if max index of the last context is the same as the max index of the second last context
    context_queue = drop_last_repeat_context(context_queue)
    logger.debug(f"去重后上下文队列数量: {len(context_queue)}")
    
    # 计算批次数
    num_context_batches = math.ceil(len(context_queue) / context_batch_size)
    logger.debug(f"计算得到批次数: {num_context_batches}")
    
    # 按批次组织全局上下文
    global_context = []
    for i_tmp in range(num_context_batches):
        batch = context_queue[i_tmp * context_batch_size : (i_tmp + 1) * context_batch_size]
        global_context.append(batch)
        logger.debug(f"添加批次 {i_tmp}，包含 {len(batch)} 个上下文")
        
    logger.info(f"全局上下文准备完成，总计 {len(global_context)} 个批次")
    return global_context