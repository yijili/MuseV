# 导入所需模块
from typing import Any, Dict  # 导入类型提示模块，用于类型注解
from torch import nn  # 导入PyTorch的神经网络模块

import logging  # 导入日志模块，用于记录函数调用信息

# 配置日志记录器
logger = logging.getLogger(__name__)  # 创建一个以当前模块命名的日志记录器
logging.basicConfig(level=logging.info)

class TextEmbExtractor(nn.Module):
    """
    文本嵌入提取器类，用于将文本转换为嵌入向量表示
    继承自PyTorch的nn.Module基类
    """
    
    def __init__(self, tokenizer, text_encoder) -> None:
        """
        初始化文本嵌入提取器
        
        Args:
            tokenizer: 分词器对象，用于将文本转换为token序列
            text_encoder: 文本编码器对象，用于将token序列编码为向量表示
        """
        super(TextEmbExtractor, self).__init__()  # 调用父类初始化方法
        self.tokenizer = tokenizer  # 保存分词器实例
        self.text_encoder = text_encoder  # 保存文本编码器实例
        logger.info("初始化TextEmbExtractor，设置tokenizer和text_encoder")  # 记录初始化日志

    def forward(
        self,
        texts,
        text_params: Dict = None,
    ):
        """
        前向传播函数，将输入文本转换为嵌入向量
        
        Args:
            texts: 输入的文本列表或单个文本字符串
            text_params: 文本编码的额外参数字典，默认为None
            
        Returns:
            embeddings: 文本的嵌入向量表示
        """
        logger.info(f"开始处理文本: {texts}")  # 记录处理文本的日志信息
        
        # 如果未提供文本参数，则初始化为空字典
        if text_params is None:
            text_params = {}
            logger.debug("未提供text_params，使用空字典作为默认参数")  # 记录参数初始化日志
            
        # 使用tokenizer对输入文本进行分词和预处理
        logger.debug(f"使用tokenizer处理文本，最大长度为{self.tokenizer.model_max_length}")  # 记录分词处理日志
        special_prompt_input = self.tokenizer(
            texts,
            max_length=self.tokenizer.model_max_length,  # 设置最大序列长度
            padding="max_length",  # 使用最大长度进行填充
            truncation=True,  # 超过最大长度时进行截断
            return_tensors="pt",  # 返回PyTorch张量格式
        )
        logger.debug("完成文本tokenization处理")  # 记录分词完成日志
        
        # 检查是否需要使用注意力掩码
        if (
            hasattr(self.text_encoder.config, "use_attention_mask")  # 检查text_encoder配置中是否有use_attention_mask属性
            and self.text_encoder.config.use_attention_mask  # 且该属性为True
        ):
            # 如果需要使用注意力掩码，则将注意力掩码移动到text_encoder所在的设备上
            logger.debug("使用注意力掩码")  # 记录使用注意力掩码的日志
            attention_mask = special_prompt_input.attention_mask.to(
                self.text_encoder.device  # 将注意力掩码移动到text_encoder所在设备
            )
        else:
            # 如果不需要使用注意力掩码，则设置为None
            attention_mask = None
            logger.debug("不使用注意力掩码")  # 记录不使用注意力掩码的日志

        # 将处理好的输入数据传递给text_encoder进行编码
        logger.debug(f"调用text_encoder进行文本编码，额外参数: {text_params}")  # 记录编码器调用日志
        embeddings = self.text_encoder(
            special_prompt_input.input_ids.to(self.text_encoder.device),  # 将input_ids移动到text_encoder所在设备
            attention_mask=attention_mask,  # 传入注意力掩码（如果有的话）
            **text_params  # 展开传入额外的文本参数
        )
        logger.info("完成文本嵌入提取")  # 记录嵌入提取完成日志
        
        return embeddings  # 返回文本嵌入向量