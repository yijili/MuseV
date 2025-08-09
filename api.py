# api.py
import os
import sys
import time
import argparse
import logging
import uuid
from typing import Optional
from datetime import datetime
import asyncio
import threading
import signal
import atexit
import gc
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image
import imageio
import glob
import pickle
import shutil
import re
from argparse import Namespace
from tqdm import tqdm
import copy

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# 添加项目根目录到 Python 路径
project_dir = os.path.abspath(os.path.dirname(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置北京时间
import time
import logging.handlers

class BeijingTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = time.localtime(record.created)
        # 转换为北京时间 (UTC+8)
        beijing_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(record.created + 8 * 3600))
        return beijing_time

# 重新设置日志格式，使用北京时间
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
# 创建新的处理器并设置北京时间格式
handler = logging.StreamHandler()
formatter = BeijingTimeFormatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 全局变量
device = None
weight_dtype = torch.float16

# 从 MuseV 中导入必要的函数和模块
try:
    from musev.pipelines.pipeline_controlnet_predictor import DiffusersPipelinePredictor
    from musev.models.referencenet_loader import load_referencenet_by_name
    from musev.models.ip_adapter_loader import (
        load_vision_clip_encoder_by_name,
        load_ip_adapter_image_proj_by_name,
    )
    from musev.models.facein_loader import load_facein_extractor_and_proj_by_name
    from musev.models.ip_adapter_face_loader import load_ip_adapter_face_extractor_and_proj_by_name
    from musev.models.unet_loader import load_unet_by_name
    from musev.models.controlnet import PoseGuider
    from musev.utils.util import save_videos_grid_with_opencv
    from mmcm.utils.load_util import load_pyhon_obj
    from mmcm.utils.seed_util import set_all_seed
    from mmcm.utils.signature import get_signature_of_string
    from mmcm.vision.utils.data_type_util import is_image, read_image_as_5d
    from mmcm.utils.str_util import clean_str_for_save
    from mmcm.vision.data.video_dataset import DecordVideoDataset
    modules_available = True
except ImportError as e:
    logger.warning(f"部分模块导入失败: {e}")
    modules_available = False

# 服务相关全局变量
server_thread = None
should_exit = False

# 任务管理相关
active_tasks = {}  # 存储正在进行的任务
task_lock = threading.Lock()  # 任务锁

class TaskStatus:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = "running"  # running, completed, failed, cancelled
        self.progress = 0
        self.message = ""
        self.result = None
        self.thread = None  # 添加线程引用

def cleanup_models():
    """清理模型以释放内存"""
    global sd_predictor, referencenet, vision_clip_extractor, ip_adapter_image_proj
    global face_emb_extractor, facein_image_proj, ip_adapter_face_emb_extractor
    global ip_adapter_face_image_proj, pose_guider, unet
    
    logger.info("清理模型内存...")
    
    # 删除模型引用
    if 'sd_predictor' in globals() and sd_predictor:
        del sd_predictor
    if 'referencenet' in globals() and referencenet:
        del referencenet
    if 'vision_clip_extractor' in globals() and vision_clip_extractor:
        del vision_clip_extractor
    if 'ip_adapter_image_proj' in globals() and ip_adapter_image_proj:
        del ip_adapter_image_proj
    if 'face_emb_extractor' in globals() and face_emb_extractor:
        del face_emb_extractor
    if 'facein_image_proj' in globals() and facein_image_proj:
        del facein_image_proj
    if 'ip_adapter_face_emb_extractor' in globals() and ip_adapter_face_emb_extractor:
        del ip_adapter_face_emb_extractor
    if 'ip_adapter_face_image_proj' in globals() and ip_adapter_face_image_proj:
        del ip_adapter_face_image_proj
    if 'pose_guider' in globals() and pose_guider:
        del pose_guider
    if 'unet' in globals() and unet:
        del unet
        
    sd_predictor = None
    referencenet = None
    vision_clip_extractor = None
    ip_adapter_image_proj = None
    face_emb_extractor = None
    facein_image_proj = None
    ip_adapter_face_emb_extractor = None
    ip_adapter_face_image_proj = None
    pose_guider = None
    unet = None
    
    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.info("模型内存清理完成")

def load_models():
    """加载所有模型"""
    global sd_predictor, referencenet, vision_clip_extractor, ip_adapter_image_proj
    global face_emb_extractor, facein_image_proj, ip_adapter_face_emb_extractor
    global ip_adapter_face_image_proj, pose_guider, unet, device, weight_dtype
    
    if not modules_available:
        logger.error("必要的模块未正确导入，无法加载模型")
        return False
    
    # 使用 GPU 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16
    
    logger.info(f"使用设备: {device}")
    logger.info("开始加载模型...")
    
    # 设置项目目录
    PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
    CHECKPOINTS_DIR = os.path.join(PROJECT_DIR, "checkpoints")
    logger.info(f"PROJECT_DIR: {PROJECT_DIR}")
    logger.info(f"CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")
    
    # 检查 checkpoints 目录是否存在
    if not os.path.exists(CHECKPOINTS_DIR):
        logger.warning(f"Checkpoints 目录不存在: {CHECKPOINTS_DIR}")
        logger.warning("请确保已按照 MuseV 部署教程下载所有模型到 checkpoints 目录")
    
    # 修改模型配置参数，确保路径正确
    args_dict = {
        "add_static_video_prompt": False,
        "context_batch_size": 1,
        "context_frames": 12,
        "context_overlap": 4,
        "context_schedule": "uniform_v2",
        "context_stride": 1,
        "controlnet_conditioning_scale": 1.0,
        "controlnet_name": "dwpose_body_hand",
        "cross_attention_dim": 768,
        "enable_zero_snr": False,
        "end_to_end": True,
        "face_image_path": None,
        "facein_model_cfg_path": os.path.join(PROJECT_DIR, "configs/model/facein.py"),
        "facein_model_name": None,
        "facein_scale": 1.0,
        "fix_condition_images": False,
        "fixed_ip_adapter_image": True,
        "fixed_refer_face_image": True,
        "fixed_refer_image": True,
        "fps": 4,
        "guidance_scale": 7.5,
        "height": None,
        "img_length_ratio": 1.0,
        "img_weight": 0.001,
        "interpolation_factor": 1,
        "ip_adapter_face_model_cfg_path": os.path.join(PROJECT_DIR, "configs/model/ip_adapter.py"),
        "ip_adapter_face_model_name": None,
        "ip_adapter_model_cfg_path": os.path.join(PROJECT_DIR, "configs/model/ip_adapter.py"),
        "ip_adapter_model_name": "musev_referencenet",
        "ip_adapter_scale": 1.0,
        "ipadapter_image_path": None,
        "lcm_model_cfg_path": os.path.join(PROJECT_DIR, "configs/model/lcm_model.py"),
        "lcm_model_name": None,
        "log_level": "INFO",
        "motion_speed": 8.0,
        "n_batch": 1,
        "n_cols": 3,
        "n_repeat": 1,
        "n_vision_condition": 1,
        "need_hist_match": False,
        "need_img_based_video_noise": True,
        "need_redraw": False,
        "need_return_condition": False,
        "need_return_videos": False,
        "need_video2video": False,
        "negative_prompt": "V2",
        "negprompt_cfg_path": os.path.join(PROJECT_DIR, "configs/model/negative_prompt.py"),
        "noise_type": "video_fusion",
        "num_inference_steps": 30,
        "output_dir": "./results/",
        "overwrite": False,
        "pose_guider_model_path": None,
        "prompt_only_use_image_prompt": False,
        "record_mid_video_latents": False,
        "record_mid_video_noises": False,
        "redraw_condition_image": False,
        "redraw_condition_image_with_facein": True,
        "redraw_condition_image_with_ip_adapter_face": True,
        "redraw_condition_image_with_ipdapter": True,
        "redraw_condition_image_with_referencenet": True,
        "referencenet_image_path": None,
        "referencenet_model_cfg_path": os.path.join(PROJECT_DIR, "configs/model/referencenet.py"),
        "referencenet_model_name": "musev_referencenet",
        "save_filetype": "mp4",
        "save_images": False,
        "sd_model_cfg_path": os.path.join(PROJECT_DIR, "configs/model/T2I_all_model.py"),
        "sd_model_name": "majicmixRealv6Fp16",
        "seed": None,
        "strength": 0.8,
        "target_datas": "boy_dance2",
        "test_data_path": os.path.join(PROJECT_DIR, "configs/tasks/example.yaml"),
        "time_size": 12,
        "unet_model_cfg_path": os.path.join(PROJECT_DIR, "configs/model/motion_model.py"),
        "unet_model_name": "musev_referencenet",
        "use_condition_image": True,
        "use_video_redraw": True,
        "vae_model_path": os.path.join(CHECKPOINTS_DIR, "vae/sd-vae-ft-mse"),
        "video_guidance_scale": 3.5,
        "video_guidance_scale_end": None,
        "video_guidance_scale_method": "linear",
        "video_has_condition": True,
        "video_is_middle": False,
        "video_negative_prompt": "V2",
        "video_num_inference_steps": 10,
        "video_overlap": 1,
        "video_strength": 1.0,
        "vision_clip_extractor_class_name": "ImageClipVisionFeatureExtractor",
        "vision_clip_model_path": os.path.join(CHECKPOINTS_DIR, "IP-Adapter/models/image_encoder"),
        "w_ind_noise": 0.5,
        "which2video": "video_middle",
        "width": None,
        "write_info": False,
    }
    
    args = Namespace(**args_dict)
    logger.setLevel(args.log_level)
    
    # 参数设置
    sd_model_cfg_path = args.sd_model_cfg_path
    sd_model_name = args.sd_model_name
    unet_model_cfg_path = args.unet_model_cfg_path
    unet_model_name = args.unet_model_name
    referencenet_model_cfg_path = args.referencenet_model_cfg_path
    referencenet_model_name = args.referencenet_model_name
    ip_adapter_model_cfg_path = args.ip_adapter_model_cfg_path
    ip_adapter_model_name = args.ip_adapter_model_name
    vision_clip_model_path = args.vision_clip_model_path
    vision_clip_extractor_class_name = args.vision_clip_extractor_class_name
    facein_model_cfg_path = args.facein_model_cfg_path
    facein_model_name = args.facein_model_name
    ip_adapter_face_model_cfg_path = args.ip_adapter_face_model_cfg_path
    ip_adapter_face_model_name = args.ip_adapter_face_model_name
    pose_guider_model_path = args.pose_guider_model_path
    vae_model_path = args.vae_model_path
    cross_attention_dim = args.cross_attention_dim
    negprompt_cfg_path = args.negprompt_cfg_path
    
    # 检查配置文件是否存在
    if not os.path.exists(sd_model_cfg_path):
        logger.error(f"SD模型配置文件不存在: {sd_model_cfg_path}")
        return False
    
    if not os.path.exists(unet_model_cfg_path):
        logger.error(f"UNet模型配置文件不存在: {unet_model_cfg_path}")
        return False
    
    if not os.path.exists(referencenet_model_cfg_path):
        logger.error(f"ReferenceNet模型配置文件不存在: {referencenet_model_cfg_path}")
        return False
    
    if not os.path.exists(ip_adapter_model_cfg_path):
        logger.error(f"IP-Adapter模型配置文件不存在: {ip_adapter_model_cfg_path}")
        return False
    
    # 加载 SD 模型参数
    try:
        sd_model_params_dict_src = load_pyhon_obj(sd_model_cfg_path, "MODEL_CFG")
        sd_model_params_dict = {
            k: v
            for k, v in sd_model_params_dict_src.items()
            if sd_model_name == "all" or k in sd_model_name
        }
        
        if len(sd_model_params_dict) == 0:
            raise ValueError(
                "没有找到目标模型，请设置以下模型之一: {}".format(
                    " ".join(list(sd_model_params_dict_src.keys()))
                )
            )
        
        logger.info("运行模型, T2I SD")
        logger.info(str(sd_model_params_dict))
    except Exception as e:
        logger.error(f"加载SD模型参数失败: {e}")
        return False
    
    # 加载 LCM 模型
    lcm_model_name = args.lcm_model_name
    lcm_model_cfg_path = args.lcm_model_cfg_path
    lcm_lora_dct = None
    if lcm_model_name is not None and os.path.exists(lcm_model_cfg_path):
        try:
            lcm_model_params_dict_src = load_pyhon_obj(lcm_model_cfg_path, "MODEL_CFG")
            logger.info("lcm_model_params_dict_src")
            lcm_lora_dct = lcm_model_params_dict_src[lcm_model_name]
        except Exception as e:
            logger.warning(f"加载LCM模型失败: {e}")
    logger.info(f"lcm: {lcm_model_name}, {lcm_lora_dct}")
    
    # 加载 motion net 参数
    try:
        if os.path.isdir(unet_model_cfg_path):
            unet_model_path = unet_model_cfg_path
        elif os.path.isfile(unet_model_cfg_path):
            unet_model_params_dict_src = load_pyhon_obj(unet_model_cfg_path, "MODEL_CFG")
            logger.info(f"unet_model_params_dict_src {list(unet_model_params_dict_src.keys())}")
            unet_model_path = unet_model_params_dict_src[unet_model_name]["unet"]
        else:
            raise ValueError(f"期望目录或文件，但给定 {unet_model_cfg_path}")
        logger.info(f"unet: {unet_model_name}, {unet_model_path}")
    except Exception as e:
        logger.error(f"加载UNet模型路径失败: {e}")
        return False
    
    # 加载 referencenet
    referencenet_model_path = None
    try:
        if referencenet_model_name is not None:
            if os.path.isdir(referencenet_model_cfg_path):
                referencenet_model_path = referencenet_model_cfg_path
            elif os.path.isfile(referencenet_model_cfg_path):
                referencenet_model_params_dict_src = load_pyhon_obj(
                    referencenet_model_cfg_path, "MODEL_CFG"
                )
                logger.info(
                    f"referencenet_model_params_dict_src {list(referencenet_model_params_dict_src.keys())}",
                )
                referencenet_model_path = referencenet_model_params_dict_src[
                    referencenet_model_name
                ]["net"]
            else:
                raise ValueError(f"期望目录或文件，但给定 {referencenet_model_cfg_path}")
        logger.info(f"referencenet: {referencenet_model_name}, {referencenet_model_path}")
    except Exception as e:
        logger.warning(f"加载ReferenceNet模型路径失败: {e}")
    
    # 加载 ip_adapter
    ip_adapter_model_params_dict = None
    try:
        if ip_adapter_model_name is not None:
            ip_adapter_model_params_dict_src = load_pyhon_obj(
                ip_adapter_model_cfg_path, "MODEL_CFG"
            )
            logger.info(f"ip_adapter_model_params_dict_src {list(ip_adapter_model_params_dict_src.keys())}")
            ip_adapter_model_params_dict = ip_adapter_model_params_dict_src[
                ip_adapter_model_name
            ]
        logger.info(f"ip_adapter: {ip_adapter_model_name}, {ip_adapter_model_params_dict}")
    except Exception as e:
        logger.warning(f"加载IP-Adapter模型参数失败: {e}")
    
    # 加载 facein
    facein_model_params_dict = None
    try:
        if facein_model_name is not None:
            facein_model_params_dict_src = load_pyhon_obj(facein_model_cfg_path, "MODEL_CFG")
            logger.info(f"facein_model_params_dict_src {list(facein_model_params_dict_src.keys())}")
            facein_model_params_dict = facein_model_params_dict_src[facein_model_name]
        logger.info(f"facein: {facein_model_name}, {facein_model_params_dict}")
    except Exception as e:
        logger.warning(f"加载FaceIn模型参数失败: {e}")
    
    # 加载 ip_adapter_face
    ip_adapter_face_model_params_dict = None
    try:
        if ip_adapter_face_model_name is not None:
            ip_adapter_face_model_params_dict_src = load_pyhon_obj(
                ip_adapter_face_model_cfg_path, "MODEL_CFG"
            )
            logger.info(
                f"ip_adapter_face_model_params_dict_src {list(ip_adapter_face_model_params_dict_src.keys())}",
            )
            ip_adapter_face_model_params_dict = ip_adapter_face_model_params_dict_src[
                ip_adapter_face_model_name
            ]
        logger.info(
            f"ip_adapter_face: {ip_adapter_face_model_name}, {ip_adapter_face_model_params_dict}"
        )
    except Exception as e:
        logger.warning(f"加载IP-Adapter Face模型参数失败: {e}")
    
    # 加载模型
    model_loaded = False
    for model_name, sd_model_params in sd_model_params_dict.items():
        try:
            lora_dict = sd_model_params.get("lora", None)
            model_sex = sd_model_params.get("sex", None)
            model_style = sd_model_params.get("style", None)
            sd_model_path = sd_model_params["sd"]
            test_model_vae_model_path = sd_model_params.get("vae", vae_model_path)
            
            logger.info(f"正在加载模型: {model_name}")
            logger.info(f"SD模型路径: {sd_model_path}")
            logger.info(f"VAE模型路径: {test_model_vae_model_path}")
            
            # 检查模型文件是否存在
            if not os.path.exists(sd_model_path):
                logger.error(f"SD模型文件不存在: {sd_model_path}")
                continue
                
            # 加载 UNet
            unet = load_unet_by_name(
                model_name=unet_model_name,
                sd_unet_model=unet_model_path,
                sd_model=sd_model_path,
                cross_attention_dim=cross_attention_dim,
                need_t2i_facein=facein_model_name is not None,
                # facein 目前没参与训练，但在unet中定义了，载入相关参数会报错，所以用strict控制
                strict=not (facein_model_name is not None),
                need_t2i_ip_adapter_face=ip_adapter_face_model_name is not None,
            )
            
            # 加载 referencenet
            if referencenet_model_name is not None and referencenet_model_path and os.path.exists(referencenet_model_path):
                referencenet = load_referencenet_by_name(
                    model_name=referencenet_model_name,
                    sd_referencenet_model=referencenet_model_path,
                    cross_attention_dim=cross_attention_dim,
                )
            else:
                referencenet = None
                referencenet_model_name = "no"
            
            # 加载 vision_clip_extractor
            if vision_clip_extractor_class_name is not None and os.path.exists(vision_clip_model_path):
                vision_clip_extractor = load_vision_clip_encoder_by_name(
                    ip_image_encoder=vision_clip_model_path,
                    vision_clip_extractor_class_name=vision_clip_extractor_class_name,
                )
                logger.info(
                    f"vision_clip_extractor, name={vision_clip_extractor_class_name}, path={vision_clip_model_path}"
                )
            else:
                vision_clip_extractor = None
                logger.info(f"vision_clip_extractor, None")
            
            # 加载 ip_adapter_image_proj
            if ip_adapter_model_name is not None and ip_adapter_model_params_dict:
                ip_image_encoder = vision_clip_model_path
                ip_ckpt = ip_adapter_model_params_dict["ip_ckpt"]

                logger.info(f"ip_image_encoder： {ip_image_encoder}")
                logger.info(f"ip_ckpt： {ip_ckpt}")
                
                if os.path.exists(ip_image_encoder) and os.path.exists(ip_ckpt):
                    ip_adapter_image_proj = load_ip_adapter_image_proj_by_name(
                        model_name=ip_adapter_model_name,
                        ip_image_encoder=ip_image_encoder,
                        ip_ckpt=ip_ckpt,
                        cross_attention_dim=cross_attention_dim,
                        clip_embeddings_dim=ip_adapter_model_params_dict["clip_embeddings_dim"],
                        clip_extra_context_tokens=ip_adapter_model_params_dict[
                            "clip_extra_context_tokens"
                        ],
                        ip_scale=ip_adapter_model_params_dict["ip_scale"],
                        device=device,
                    )
                else:
                    ip_adapter_image_proj = None
                    logger.warning(f"IP-Adapter模型文件不存在: image_encoder={ip_image_encoder}, ckpt={ip_ckpt}")
            else:
                ip_adapter_image_proj = None
                ip_adapter_model_name = "no"
            
            # 加载 facein 模块
            if facein_model_name is not None and facein_model_params_dict:
                ip_image_encoder = facein_model_params_dict["ip_image_encoder"]
                ip_ckpt = facein_model_params_dict["ip_ckpt"]
                
                if os.path.exists(ip_image_encoder) and os.path.exists(ip_ckpt):
                    (
                        face_emb_extractor,
                        facein_image_proj,
                    ) = load_facein_extractor_and_proj_by_name(
                        model_name=facein_model_name,
                        ip_image_encoder=ip_image_encoder,
                        ip_ckpt=ip_ckpt,
                        cross_attention_dim=cross_attention_dim,
                        clip_embeddings_dim=facein_model_params_dict["clip_embeddings_dim"],
                        clip_extra_context_tokens=facein_model_params_dict[
                            "clip_extra_context_tokens"
                        ],
                        ip_scale=facein_model_params_dict["ip_scale"],
                        device=device,
                        unet=unet,
                    )
                else:
                    face_emb_extractor = None
                    facein_image_proj = None
                    logger.warning(f"FaceIn模型文件不存在: image_encoder={ip_image_encoder}, ckpt={ip_ckpt}")
            else:
                face_emb_extractor = None
                facein_image_proj = None
            
            # 加载 ip_adapter_face 模块
            if ip_adapter_face_model_name is not None and ip_adapter_face_model_params_dict:
                ip_image_encoder = ip_adapter_face_model_params_dict["ip_image_encoder"]
                ip_ckpt = ip_adapter_face_model_params_dict["ip_ckpt"]
                
                if os.path.exists(ip_image_encoder) and os.path.exists(ip_ckpt):
                    (
                        ip_adapter_face_emb_extractor,
                        ip_adapter_face_image_proj,
                    ) = load_ip_adapter_face_extractor_and_proj_by_name(
                        model_name=ip_adapter_face_model_name,
                        ip_image_encoder=ip_image_encoder,
                        ip_ckpt=ip_ckpt,
                        cross_attention_dim=cross_attention_dim,
                        clip_embeddings_dim=ip_adapter_face_model_params_dict[
                            "clip_embeddings_dim"
                        ],
                        clip_extra_context_tokens=ip_adapter_face_model_params_dict[
                            "clip_extra_context_tokens"
                        ],
                        ip_scale=ip_adapter_face_model_params_dict["ip_scale"],
                        device=device,
                        unet=unet,
                    )
                else:
                    ip_adapter_face_emb_extractor = None
                    ip_adapter_face_image_proj = None
                    logger.warning(f"IP-Adapter Face模型文件不存在: image_encoder={ip_image_encoder}, ckpt={ip_ckpt}")
            else:
                ip_adapter_face_emb_extractor = None
                ip_adapter_face_image_proj = None
            
            # 加载 pose_guider
            if pose_guider_model_path is not None and os.path.exists(pose_guider_model_path):
                logger.info(f"PoseGuider ={pose_guider_model_path}")
                pose_guider = PoseGuider.from_pretrained(
                    pose_guider_model_path,
                    conditioning_embedding_channels=320,
                    block_out_channels=(16, 32, 96, 256),
                )
            else:
                pose_guider = None
            
            logger.info(f"test_model_vae_model_path {test_model_vae_model_path}")
            
            # 创建预测器
            sd_predictor = DiffusersPipelinePredictor(
                sd_model_path=sd_model_path,
                unet=unet,
                lora_dict=lora_dict,
                lcm_lora_dct=lcm_lora_dct,
                device=device,
                dtype=weight_dtype,
                negative_embedding=[
                    [os.path.join(CHECKPOINTS_DIR, "embedding/badhandv4.pt"), "badhandv4"],
                    [
                        os.path.join(CHECKPOINTS_DIR, "embedding/ng_deepnegative_v1_75t.pt"),
                        "ng_deepnegative_v1_75t",
                    ],
                    [
                        os.path.join(CHECKPOINTS_DIR, "embedding/EasyNegativeV2.safetensors"),
                        "EasyNegativeV2",
                    ],
                    [
                        os.path.join(CHECKPOINTS_DIR, "embedding/bad_prompt_version2-neg.pt"),
                        "bad_prompt_version2-neg",
                    ],
                ],
                referencenet=referencenet,
                ip_adapter_image_proj=ip_adapter_image_proj,
                vision_clip_extractor=vision_clip_extractor,
                facein_image_proj=facein_image_proj,
                face_emb_extractor=face_emb_extractor,
                vae_model=test_model_vae_model_path,
                ip_adapter_face_emb_extractor=ip_adapter_face_emb_extractor,
                ip_adapter_face_image_proj=ip_adapter_face_image_proj,
                pose_guider=pose_guider,
                controlnet_name=args.controlnet_name,
                include_body=True,
                include_face=False,
                include_hand=True,
                enable_zero_snr=args.enable_zero_snr,
            )
            logger.debug(f"加载 referencenet")
            model_loaded = True
            break  # 只加载第一个模型
        except Exception as e:
            logger.error(f"加载模型 {model_name} 失败: {e}")
            continue
    
    if not model_loaded:
        logger.error("未能成功加载任何模型")
        return False
    
    logger.info("模型加载完成")
    if torch.cuda.is_available():
        logger.info(f"GPU内存状态: 已分配 {torch.cuda.memory_allocated()/1024**2:.1f} MB, 已缓存 {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    
    return True

# 注册退出处理函数
atexit.register(cleanup_models)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时不加载模型
    logger.info("服务启动，模型将在需要时加载")
    yield
    # 关闭时的清理代码
    logger.info("正在关闭应用...")
    cleanup_models()
    logger.info("应用已关闭")

app = FastAPI(title="MuseV API", description="MuseV FastAPI 服务，提供 text2video 和 video2video 接口", lifespan=lifespan)

class TextToVideoRequest(BaseModel):
    prompt: str
    image_path: str
    seed: int = -1
    fps: int = 4
    width: int = -1
    height: int = -1
    video_length: int = 12
    img_edge_ratio: float = 1.0
    output_path: Optional[str] = None
    task_id: Optional[str] = None

class VideoToVideoRequest(BaseModel):
    prompt: str
    image_path: str
    video_path: str
    processor: str = "dwpose_body_hand"
    seed: int = -1
    fps: int = 4
    width: int = -1
    height: int = -1
    video_length: int = 12
    img_edge_ratio: float = 1.0
    output_path: Optional[str] = None
    task_id: Optional[str] = None

class InferenceResponse(BaseModel):
    task_id: str
    message: str
    video_path: Optional[str] = None

class TaskStatusRequest(BaseModel):
    task_id: str

class TaskResultRequest(BaseModel):
    task_id: str

class CancelTaskRequest(BaseModel):
    task_id: str

def generate_unique_filename(extension: str) -> str:
    """生成带时间戳的唯一文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}.{extension}"

def check_task_cancelled(task_id: str) -> bool:
    """检查任务是否被取消"""
    with task_lock:
        if task_id in active_tasks:
            return active_tasks[task_id].status == "cancelled"
    return False

def limit_shape(image, input_w, input_h, img_edge_ratio, max_image_edge=1280):
    """限制生成视频形状以避免GPU内存溢出"""
    if input_h == -1 and input_w == -1:
        if isinstance(image, np.ndarray):
            input_h, input_w, _ = image.shape
        elif isinstance(image, Image.Image):
            input_w, input_h = image.size
        else:
            raise ValueError(
                f"image should be in [image, ndarray], but given {type(image)}"
            )
    if img_edge_ratio == 0:
        img_edge_ratio = 1
    img_edge_ratio_infact = min(max_image_edge / max(input_h, input_w), img_edge_ratio)
    if img_edge_ratio != 1:
        return (
            img_edge_ratio_infact,
            input_w * img_edge_ratio_infact,
            input_h * img_edge_ratio_infact,
        )
    else:
        return img_edge_ratio_infact, -1, -1

def limit_length(length):
    """限制生成视频帧数以避免GPU内存溢出"""
    if length > 24 * 6:
        length = 24 * 6
    return length

def read_image(path):
    """读取图像"""
    name = os.path.basename(path).split(".")[0]
    image = read_image_as_5d(path)
    return image, name

def read_image_lst(path):
    """读取图像列表"""
    images_names = [read_image(x) for x in path]
    images, names = zip(*images_names)
    images = np.concatenate(images, axis=2)
    name = "_".join(names)
    return images, name

def read_image_and_name(path):
    """读取图像和名称"""
    if isinstance(path, str):
        path = [path]
    images, name = read_image_lst(path)
    return images, name

def generate_cuid():
    """生成唯一ID"""
    try:
        import cuid
        return cuid.cuid()
    except ImportError:
        return str(uuid.uuid4())

def text_to_video_api(
    task_id: str,
    prompt: str,
    image_path: str,
    seed: int,
    fps: int,
    width: int,
    height: int,
    video_length: int,
    img_edge_ratio: float,
    output_path: Optional[str] = None
) -> dict:
    """API 版本的 text_to_video 函数，支持任务中断"""
    try:
        logger.info(f"开始执行 text_to_video，任务ID: {task_id}，图像路径: {image_path}")
        
        # 加载模型
        if not load_models():
            raise HTTPException(status_code=500, detail="模型加载失败")
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            cleanup_models()  # 释放模型
            return {"message": "任务已被取消"}
        
        # 设置参数
        output_dir = output_path if output_path else './results'
        os.makedirs(output_dir, exist_ok=True)
        
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 10
                active_tasks[task_id].message = "处理输入图像..."
        
        # 读取图像
        try:
            image = Image.open(image_path)
            image_np = np.array(image)
        except Exception as e:
            logger.error(f"读取图像失败: {e}")
            cleanup_models()  # 释放模型
            raise HTTPException(status_code=500, detail=f"读取图像失败: {e}")
        
        # 限制形状
        try:
            img_edge_ratio_infact, out_w, out_h = limit_shape(
                image_np, width, height, img_edge_ratio
            )
        except Exception as e:
            logger.error(f"限制图像形状失败: {e}")
            cleanup_models()  # 释放模型
            raise HTTPException(status_code=500, detail=f"限制图像形状失败: {e}")
        
        if out_w != -1 and out_h != -1:
            width, height = int(out_w), int(out_h)
        
        # 限制长度
        video_length = limit_length(video_length)
        
        # 生成文件名
        image_cuid = generate_cuid()
        time_size = int(video_length)
        
        test_data = {
            "name": image_cuid,
            "prompt": prompt,
            "condition_images": image_path,
            "refer_image": image_path,
            "ipadapter_image": image_path,
            "height": height,
            "width": width,
            "img_length_ratio": img_edge_ratio_infact,
        }
        
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 20
                active_tasks[task_id].message = "准备生成参数..."
        
        # 准备参数，与gradio_text2video.py保持一致
        test_data_name = test_data.get("name", test_data)
        prompt = test_data["prompt"]
        prefix_prompt = ""
        suffix_prompt = ", beautiful, masterpiece, best quality"
        prompt = prefix_prompt + prompt + suffix_prompt
        prompt_hash = get_signature_of_string(prompt, length=5)
        test_data["prompt_hash"] = prompt_hash
        test_data_height = test_data.get("height", height)
        test_data_width = test_data.get("width", width)
        test_data_condition_images_path = test_data.get("condition_images", None)
        test_data_condition_images_index = test_data.get("condition_images_index", None)
        test_data_redraw_condition_image = test_data.get("redraw_condition_image", False)
        
        # 读取条件图像
        test_data_condition_images = None
        test_data_condition_images_name = "no"
        condition_image_height = None
        condition_image_width = None
        
        if (
            test_data_condition_images_path is not None
            and (
                isinstance(test_data_condition_images_path, list)
                or (
                    isinstance(test_data_condition_images_path, str)
                    and is_image(test_data_condition_images_path)
                )
            )
        ):
            try:
                (
                    test_data_condition_images,
                    test_data_condition_images_name,
                ) = read_image_and_name(test_data_condition_images_path)
                condition_image_height = test_data_condition_images.shape[3]
                condition_image_width = test_data_condition_images.shape[4]
            except Exception as e:
                logger.warning(f"读取条件图像失败: {e}")
        
        # 当没有指定生成视频的宽高时，使用输入条件的宽高
        if test_data_height in [None, -1]:
            test_data_height = condition_image_height

        if test_data_width in [None, -1]:
            test_data_width = condition_image_width

        test_data_img_length_ratio = float(
            test_data.get("img_length_ratio", img_edge_ratio_infact)
        )
        # 为了和video2video保持对齐，使用64而不是8作为宽、高最小粒度
        test_data_height = int(test_data_height * test_data_img_length_ratio // 64 * 64)
        test_data_width = int(test_data_width * test_data_img_length_ratio // 64 * 64)
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            cleanup_models()  # 释放模型
            return {"message": "任务已被取消"}
        
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 30
                active_tasks[task_id].message = "生成视频..."
        
        # 生成输出路径
        save_file_name = (
            f"t2v_musev_case={test_data_name}"
            f"_w={test_data_width}_h={test_data_height}_t={time_size}"
            f"_s={seed}_p={prompt_hash}"
        )
        save_file_name = clean_str_for_save(save_file_name)
        output_video_path = os.path.join(
            output_dir,
            f"{save_file_name}.mp4",
        )
        
        logger.info("正在生成视频...")
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            cleanup_models()  # 释放模型
            return {"message": "任务已被取消"}
        
        # 实际调用模型进行推理
        test_data_seed = np.random.randint(0, 1e8) if seed in [None, -1] else seed
        cpu_generator, gpu_generator = set_all_seed(int(test_data_seed))
        
        # 调用模型推理，与gradio_text2video.py保持一致
        try:
            out_videos = sd_predictor.run_pipe_text2video(
                video_length=time_size,
                prompt=prompt,
                width=test_data_width,
                height=test_data_height,
                generator=gpu_generator,
                noise_type="video_fusion",
                negative_prompt="V2",
                video_negative_prompt="V2",
                max_batch_num=1,
                strength=0.8,
                need_img_based_video_noise=True,
                video_num_inference_steps=10,
                condition_images=test_data_condition_images,
                fix_condition_images=False,
                video_guidance_scale=3.5,
                guidance_scale=7.5,
                num_inference_steps=30,
                redraw_condition_image=False,
                img_weight=0.001,
                w_ind_noise=0.5,
                n_vision_condition=1,
                motion_speed=8.0,
                need_hist_match=False,
                video_guidance_scale_end=None,
                video_guidance_scale_method="linear",
                vision_condition_latent_index=test_data_condition_images_index,
                refer_image=test_data_condition_images,
                fixed_refer_image=True,
                redraw_condition_image_with_referencenet=True,
                ip_adapter_image=test_data_condition_images,
                refer_face_image=None,
                fixed_refer_face_image=True,
                facein_scale=1.0,
                redraw_condition_image_with_facein=True,
                ip_adapter_face_scale=1.0,
                redraw_condition_image_with_ip_adapter_face=True,
                fixed_ip_adapter_image=True,
                ip_adapter_scale=1.0,
                redraw_condition_image_with_ipdapter=True,
                prompt_only_use_image_prompt=False,
                # serial_denoise parameter start
                record_mid_video_noises=False,
                record_mid_video_latents=False,
                video_overlap=1,
                # serial_denoise parameter end
                # parallel_denoise parameter start
                context_schedule="uniform_v2",
                context_frames=12,
                context_stride=1,
                context_overlap=4,
                context_batch_size=1,
                interpolation_factor=1,
                # parallel_denoise parameter end
            )
        except Exception as e:
            logger.error(f"模型推理失败: {e}")
            cleanup_models()  # 释放模型
            raise HTTPException(status_code=500, detail=f"模型推理失败: {e}")
        
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 80
                active_tasks[task_id].message = "保存视频..."
        
        # 保存视频
        try:
            out = np.concatenate([out_videos], axis=0)
            texts = ["out"]
            save_videos_grid_with_opencv(
                out,
                output_video_path,
                texts=texts,
                fps=fps,
                tensor_order="b c t h w",
                n_cols=3,
                write_info=False,
                save_filetype="mp4",
                save_images=False,
            )
        except Exception as e:
            logger.error(f"保存视频失败: {e}")
            cleanup_models()  # 释放模型
            raise HTTPException(status_code=500, detail=f"保存视频失败: {e}")
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            cleanup_models()  # 释放模型
            return {"message": "任务已被取消"}
        
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 100
                active_tasks[task_id].status = "completed"
                active_tasks[task_id].message = "任务完成"
                active_tasks[task_id].result = {
                    "video_path": os.path.abspath(output_video_path)
                }
        
        logger.info(f"视频已保存到: {output_video_path}")
        
        # 释放模型
        cleanup_models()
        
        return {
            "video_path": os.path.abspath(output_video_path)
        }
        
    except HTTPException:
        # 重新抛出HTTP异常
        cleanup_models()  # 释放模型
        raise
    except Exception as e:
        # 更新任务状态为失败
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].status = "failed"
                active_tasks[task_id].message = f"执行出错: {str(e)}"
        logger.error(f"text_to_video 执行出错: {str(e)}")
        cleanup_models()  # 释放模型
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def video_to_video_api(
    task_id: str,
    prompt: str,
    image_path: str,
    video_path: str,
    processor: str,
    seed: int,
    fps: int,
    width: int,
    height: int,
    video_length: int,
    img_edge_ratio: float,
    output_path: Optional[str] = None
) -> dict:
    """API 版本的 video_to_video 函数，支持任务中断"""
    try:
        logger.info(f"开始执行 video_to_video，任务ID: {task_id}，图像路径: {image_path}，视频路径: {video_path}")
        
        # 加载模型
        if not load_models():
            raise HTTPException(status_code=500, detail="模型加载失败")
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            cleanup_models()  # 释放模型
            return {"message": "任务已被取消"}
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            cleanup_models()  # 释放模型
            raise HTTPException(status_code=404, detail=f"视频文件不存在: {video_path}")
        
        # 设置参数
        output_dir = output_path if output_path else './results'
        os.makedirs(output_dir, exist_ok=True)
        
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 10
                active_tasks[task_id].message = "处理输入图像和视频..."
        
        # 读取图像
        try:
            image = Image.open(image_path)
            image_np = np.array(image)
        except Exception as e:
            logger.error(f"读取图像失败: {e}")
            cleanup_models()  # 释放模型
            raise HTTPException(status_code=500, detail=f"读取图像失败: {e}")
        
        # 限制形状
        try:
            img_edge_ratio_infact, out_w, out_h = limit_shape(
                image_np, width, height, img_edge_ratio
            )
        except Exception as e:
            logger.error(f"限制图像形状失败: {e}")
            cleanup_models()  # 释放模型
            raise HTTPException(status_code=500, detail=f"限制图像形状失败: {e}")
        
        if out_w != -1 and out_h != -1:
            width, height = int(out_w), int(out_h)
        
        # 限制长度
        video_length = limit_length(video_length)
        
        # 生成文件名
        image_cuid = generate_cuid()
        time_size = int(video_length)
        
        test_data = {
            "name": image_cuid,
            "prompt": prompt,
            "video_path": video_path,
            "condition_images": image_path,
            "refer_image": image_path,
            "ipadapter_image": image_path,
            "height": height,
            "width": width,
            "img_length_ratio": img_edge_ratio_infact,
        }
        
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 20
                active_tasks[task_id].message = "准备生成参数..."
        
        # 准备参数，与gradio_video2video.py保持一致
        test_data_name = test_data.get("name", test_data)
        prompt = test_data["prompt"]
        prefix_prompt = ""
        suffix_prompt = ", beautiful, masterpiece, best quality"
        prompt = prefix_prompt + prompt + suffix_prompt
        prompt_hash = get_signature_of_string(prompt, length=5)
        test_data["prompt_hash"] = prompt_hash
        test_data_height = test_data.get("height", height)
        test_data_width = test_data.get("width", width)
        test_data_condition_images_path = test_data.get("condition_images", None)
        test_data_condition_images_index = test_data.get("condition_images_index", None)
        test_data_redraw_condition_image = test_data.get("redraw_condition_image", False)
        
        # 读取条件图像
        test_data_condition_images = None
        test_data_condition_images_name = "no"
        condition_image_height = None
        condition_image_width = None
        
        if (
            test_data_condition_images_path is not None
            and (
                isinstance(test_data_condition_images_path, list)
                or (
                    isinstance(test_data_condition_images_path, str)
                    and is_image(test_data_condition_images_path)
                )
            )
        ):
            try:
                (
                    test_data_condition_images,
                    test_data_condition_images_name,
                ) = read_image_and_name(test_data_condition_images_path)
                condition_image_height = test_data_condition_images.shape[3]
                condition_image_width = test_data_condition_images.shape[4]
            except Exception as e:
                logger.warning(f"读取条件图像失败: {e}")
        
        # 当没有指定生成视频的宽高时，使用输入条件的宽高
        if test_data_height in [None, -1]:
            test_data_height = condition_image_height

        if test_data_width in [None, -1]:
            test_data_width = condition_image_width

        test_data_img_length_ratio = float(
            test_data.get("img_length_ratio", img_edge_ratio_infact)
        )
        # 为了和video2video保持对齐，使用64而不是8作为宽、高最小粒度
        test_data_height = int(test_data_height * test_data_img_length_ratio // 64 * 64)
        test_data_width = int(test_data_width * test_data_img_length_ratio // 64 * 64)
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            cleanup_models()  # 释放模型
            return {"message": "任务已被取消"}
        
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 30
                active_tasks[task_id].message = "生成视频..."
        
        # 生成输出路径
        save_file_name = (
            f"v2v_musev_case={test_data_name}"
            f"_w={test_data_width}_h={test_data_height}_t={time_size}"
            f"_s={seed}_p={prompt_hash}"
        )
        save_file_name = clean_str_for_save(save_file_name)
        output_video_path = os.path.join(
            output_dir,
            f"{save_file_name}.mp4",
        )
        
        logger.info("正在生成视频...")
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            cleanup_models()  # 释放模型
            return {"message": "任务已被取消"}
        
        # 实际调用模型进行推理
        test_data_seed = np.random.randint(0, 1e8) if seed in [None, -1] else seed
        cpu_generator, gpu_generator = set_all_seed(int(test_data_seed))
        
        # 调用模型推理，与gradio_video2video.py保持一致
        try:
            (
                out_videos,
                out_condition,
                videos,
            ) = sd_predictor.run_pipe_video2video(
                video=video_path,
                time_size=time_size,
                step=time_size,
                sample_rate=1,
                need_return_videos=False,
                need_return_condition=False,
                controlnet_conditioning_scale=1.0,
                control_guidance_start=0.0,
                control_guidance_end=1.0,
                end_to_end=True,
                need_video2video=False,
                video_strength=1.0,
                prompt=prompt,
                width=test_data_width,
                height=test_data_height,
                generator=gpu_generator,
                noise_type="video_fusion",
                negative_prompt="V2",
                video_negative_prompt="V2",
                max_batch_num=1,
                strength=0.8,
                need_img_based_video_noise=True,
                video_num_inference_steps=10,
                condition_images=test_data_condition_images,
                fix_condition_images=False,
                video_guidance_scale=3.5,
                guidance_scale=7.5,
                num_inference_steps=30,
                redraw_condition_image=False,
                img_weight=0.001,
                w_ind_noise=0.5,
                n_vision_condition=1,
                motion_speed=8.0,
                need_hist_match=False,
                video_guidance_scale_end=None,
                video_guidance_scale_method="linear",
                vision_condition_latent_index=test_data_condition_images_index,
                refer_image=test_data_condition_images,
                fixed_refer_image=True,
                redraw_condition_image_with_referencenet=True,
                ip_adapter_image=test_data_condition_images,
                refer_face_image=None,
                fixed_refer_face_image=True,
                facein_scale=1.0,
                redraw_condition_image_with_facein=True,
                ip_adapter_face_scale=1.0,
                redraw_condition_image_with_ip_adapter_face=True,
                fixed_ip_adapter_image=True,
                ip_adapter_scale=1.0,
                redraw_condition_image_with_ipdapter=True,
                prompt_only_use_image_prompt=False,
                controlnet_processor_params={
                    "detect_resolution": min(test_data_height, test_data_width),
                    "image_resolution": min(test_data_height, test_data_width),
                },
                # serial_denoise parameter start
                record_mid_video_noises=False,
                record_mid_video_latents=False,
                video_overlap=1,
                # serial_denoise parameter end
                # parallel_denoise parameter start
                context_schedule="uniform_v2",
                context_frames=12,
                context_stride=1,
                context_overlap=4,
                context_batch_size=1,
                interpolation_factor=1,
                # parallel_denoise parameter end
                video_is_middle=False,
                video_has_condition=True,
            )
        except Exception as e:
            logger.error(f"模型推理失败: {e}")
            cleanup_models()  # 释放模型
            raise HTTPException(status_code=500, detail=f"模型推理失败: {e}")
        
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 80
                active_tasks[task_id].message = "保存视频..."
        
        # 保存视频
        try:
            batch = [out_videos]
            texts = ["out"]
            if videos is not None:
                batch.insert(0, videos / 255.0)
                texts.insert(0, "videos")
            out = np.concatenate(batch, axis=0)
            save_videos_grid_with_opencv(
                out,
                output_video_path,
                texts=texts,
                fps=fps,
                tensor_order="b c t h w",
                n_cols=3,
                write_info=False,
                save_filetype="mp4",
                save_images=False,
            )
        except Exception as e:
            logger.error(f"保存视频失败: {e}")
            cleanup_models()  # 释放模型
            raise HTTPException(status_code=500, detail=f"保存视频失败: {e}")
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            cleanup_models()  # 释放模型
            return {"message": "任务已被取消"}
        
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 100
                active_tasks[task_id].status = "completed"
                active_tasks[task_id].message = "任务完成"
                active_tasks[task_id].result = {
                    "video_path": os.path.abspath(output_video_path)
                }
        
        logger.info(f"视频已保存到: {output_video_path}")
        
        # 释放模型
        cleanup_models()
        
        return {
            "video_path": os.path.abspath(output_video_path)
        }
        
    except HTTPException:
        # 重新抛出HTTP异常
        cleanup_models()  # 释放模型
        raise
    except Exception as e:
        # 更新任务状态为失败
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].status = "failed"
                active_tasks[task_id].message = f"执行出错: {str(e)}"
        logger.error(f"video_to_video 执行出错: {str(e)}")
        cleanup_models()  # 释放模型
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_text_to_video_background(task_id: str, request: TextToVideoRequest):
    """在后台线程中运行text_to_video任务"""
    try:
        result = text_to_video_api(
            task_id=task_id,
            prompt=request.prompt,
            image_path=request.image_path,
            seed=request.seed,
            fps=request.fps,
            width=request.width,
            height=request.height,
            video_length=request.video_length,
            img_edge_ratio=request.img_edge_ratio,
            output_path=request.output_path
        )
        logger.info(f"任务 {task_id} text_to_video完成")
        
        # 更新任务状态为完成
        with task_lock:
            if task_id in active_tasks:
                if "message" in result and result["message"] == "任务已被取消":
                    active_tasks[task_id].status = "cancelled"
                    active_tasks[task_id].message = "任务已完成取消"
                else:
                    active_tasks[task_id].status = "completed"
                    active_tasks[task_id].result = result
    except Exception as e:
        # 更新任务状态为失败
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].status = "failed"
                active_tasks[task_id].message = f"执行出错: {str(e)}"
        
        logger.error(f"处理任务 {task_id} 时出错: {str(e)}")
    finally:
        # 任务完成后强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_video_to_video_background(task_id: str, request: VideoToVideoRequest):
    """在后台线程中运行video_to_video任务"""
    try:
        result = video_to_video_api(
            task_id=task_id,
            prompt=request.prompt,
            image_path=request.image_path,
            video_path=request.video_path,
            processor=request.processor,
            seed=request.seed,
            fps=request.fps,
            width=request.width,
            height=request.height,
            video_length=request.video_length,
            img_edge_ratio=request.img_edge_ratio,
            output_path=request.output_path
        )
        logger.info(f"任务 {task_id} video_to_video完成")
        
        # 更新任务状态为完成
        with task_lock:
            if task_id in active_tasks:
                if "message" in result and result["message"] == "任务已被取消":
                    active_tasks[task_id].status = "cancelled"
                    active_tasks[task_id].message = "任务已完成取消"
                else:
                    active_tasks[task_id].status = "completed"
                    active_tasks[task_id].result = result
    except Exception as e:
        # 更新任务状态为失败
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].status = "failed"
                active_tasks[task_id].message = f"执行出错: {str(e)}"
        
        logger.error(f"处理任务 {task_id} 时出错: {str(e)}")
    finally:
        # 任务完成后强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.post("/text_to_video", response_model=InferenceResponse, summary="执行文本到视频生成")
async def text_to_video_endpoint(request: TextToVideoRequest):
    """
    执行文本到视频生成
    """
    logger.info("收到 text_to_video 请求")
    
    # 检查图像文件是否存在
    if not os.path.exists(request.image_path):
        logger.error(f"图像文件不存在: {request.image_path}")
        raise HTTPException(status_code=404, detail=f"图像文件不存在: {request.image_path}")
    
    # 如果请求中包含task_id，则将其添加到active_tasks中
    task_id = request.task_id
    if not task_id:
        # 如果没有提供task_id，则生成一个新的
        task_id = str(uuid.uuid4())
    
    # 创建任务状态对象
    task_status = TaskStatus(task_id)
    
    # 将任务添加到活动任务列表
    with task_lock:
        active_tasks[task_id] = task_status
    
    # 在后台线程中运行推理任务
    thread = threading.Thread(target=run_text_to_video_background, args=(task_id, request))
    thread.start()
    
    # 保存线程引用以便可能的管理
    with task_lock:
        active_tasks[task_id].thread = thread
    
    logger.info(f"任务 {task_id} 已启动后台text_to_video线程")
    
    # 立即返回响应
    return InferenceResponse(
        task_id=task_id,
        message="任务已启动"
    )

@app.post("/video_to_video", response_model=InferenceResponse, summary="执行视频到视频生成")
async def video_to_video_endpoint(request: VideoToVideoRequest):
    """
    执行视频到视频生成
    """
    logger.info("收到 video_to_video 请求")
    
    # 检查图像和视频文件是否存在
    if not os.path.exists(request.image_path):
        logger.error(f"图像文件不存在: {request.image_path}")
        raise HTTPException(status_code=404, detail=f"图像文件不存在: {request.image_path}")
    
    if not os.path.exists(request.video_path):
        logger.error(f"视频文件不存在: {request.video_path}")
        raise HTTPException(status_code=404, detail=f"视频文件不存在: {request.video_path}")
    
    # 如果请求中包含task_id，则将其添加到active_tasks中
    task_id = request.task_id
    if not task_id:
        # 如果没有提供task_id，则生成一个新的
        task_id = str(uuid.uuid4())
    
    # 创建任务状态对象
    task_status = TaskStatus(task_id)
    
    # 将任务添加到活动任务列表
    with task_lock:
        active_tasks[task_id] = task_status
    
    # 在后台线程中运行推理任务
    thread = threading.Thread(target=run_video_to_video_background, args=(task_id, request))
    thread.start()
    
    # 保存线程引用以便可能的管理
    with task_lock:
        active_tasks[task_id].thread = thread
    
    logger.info(f"任务 {task_id} 已启动后台video_to_video线程")
    
    # 立即返回响应
    return InferenceResponse(
        task_id=task_id,
        message="任务已启动"
    )

@app.get("/task_status/{task_id}", summary="获取任务状态")
async def get_task_status(task_id: str):
    """
    获取指定任务的当前状态和进度
    """
    with task_lock:
        if task_id in active_tasks:
            task = active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task.status,
                "progress": task.progress,
                "message": task.message,
                "result": task.result
            }
        else:
            raise HTTPException(status_code=404, detail="任务不存在")

@app.post("/task_result", summary="等待并获取任务结果")
async def get_task_result(request: TaskResultRequest):
    """
    等待任务完成并返回最终结果
    """
    task_id = request.task_id
    logger.info(f"收到获取任务 {task_id} 结果的请求")
    
    # 首先检查任务是否存在
    with task_lock:
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="任务不存在")
    
    # 等待任务完成
    while True:
        with task_lock:
            if task_id in active_tasks:
                task = active_tasks[task_id]
                # 如果任务已完成、失败或被取消，则退出循环
                if task.status in ["completed", "failed", "cancelled"]:
                    break
            else:
                raise HTTPException(status_code=404, detail="任务不存在")
        
        # 等待一段时间再检查
        await asyncio.sleep(1)
    
    # 返回任务结果
    with task_lock:
        task = active_tasks[task_id]
        if task.status == "completed":
            logger.info(f"任务 {task_id} 已完成，返回结果")
            return {
                "task_id": task_id,
                "status": task.status,
                "message": task.message,
                "result": task.result
            }
        elif task.status == "failed":
            logger.info(f"任务 {task_id} 执行失败")
            raise HTTPException(status_code=500, detail=task.message)
        elif task.status == "cancelled":
            logger.info(f"任务 {task_id} 已被取消")
            raise HTTPException(status_code=499, detail="任务已被取消")  # 499表示客户端关闭请求
        else:
            raise HTTPException(status_code=500, detail="未知任务状态")

@app.post("/cancel_task", summary="取消任务")
async def cancel_task(request: CancelTaskRequest):
    """
    取消指定的任务，并等待取消完成
    """
    task_id = request.task_id
    logger.info(f"收到取消任务 {task_id} 的请求")
    
    # 首先检查任务是否存在
    with task_lock:
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="任务不存在")
    
    # 标记任务为取消状态
    with task_lock:
        active_tasks[task_id].status = "cancelled"
        active_tasks[task_id].message = "任务已被取消"
    
    # 等待任务真正完成取消
    max_wait_time = 30  # 最大等待时间30秒
    wait_interval = 0.5 # 每次等待0.5秒
    waited_time = 0
    
    while waited_time < max_wait_time:
        with task_lock:
            if task_id in active_tasks:
                task = active_tasks[task_id]
                # 如果任务已取消完成，则退出循环
                if task.status == "cancelled" and task.message == "任务已完成取消":
                    break
            else:
                # 任务已被完全清理
                break
        
        # 等待一段时间再检查
        await asyncio.sleep(wait_interval)
        waited_time += wait_interval
    
    logger.info(f"任务 {task_id} 取消完成")
    return {"message": "任务取消完成"}

@app.get("/", summary="API 根路径")
async def root():
    return {"message": "MuseV API 服务正在运行", 
            "endpoints": [
                "/text_to_video", 
                "/video_to_video",
                "/task_status/{task_id}", 
                "/task_result",
                "/cancel_task"
            ]}

@app.get("/results/{file_path:path}", summary="获取结果文件")
async def get_result_file(file_path: str):
    """
    提供对结果文件的访问
    """
    file_full_path = os.path.join("./results", file_path)
    if os.path.exists(file_full_path):
        return FileResponse(file_full_path)
    else:
        raise HTTPException(status_code=404, detail="文件未找到")

def signal_handler(signum, frame):
    """信号处理函数"""
    logger.info(f"收到信号 {signum}，正在关闭服务...")
    global should_exit
    should_exit = True
    cleanup_models()
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="主机地址")
    parser.add_argument("--port", type=int, default=7870, help="端口号")
    parser.add_argument("--isdebug", action="store_true", help="是否输出调试日志")
    parser.add_argument("--use_cpu", action="store_true", help="是否强制使用CPU")
    
    args = parser.parse_args()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 根据参数决定设备
    if args.use_cpu:
        device = torch.device("cpu")
        logger.info("强制使用CPU运行")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
    
    # 根据 isdebug 参数设置日志级别
    if args.isdebug:
        logger.setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    logger.info(f"启动 MuseV API 服务，当前版本1.0.1，主机: {args.host}，端口: {args.port}")
    
    # 不使用 reload 参数，避免重复导入
    uvicorn.run("api:app", host=args.host, port=args.port, reload=False)
