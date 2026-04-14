# -*- coding: utf-8 -*- # 指定文件编码为 UTF-8
# =============================================
# 1. 导入所需库
# =============================================
import os # 导入操作系统接口模块
import json # 导入 JSON 数据处理模块
import time # 导入时间相关功能模块
import asyncio # 导入异步 I/O 框架模块
from typing import Optional, List # 从 typing 模块导入类型提示 Optional 和 List

import torch # 导入 PyTorch 深度学习框架
import numpy as np # 导入 NumPy 用于数值计算
from fastapi import FastAPI, Response, Request as FastAPIRequest, HTTPException # 从 FastAPI 导入核心类、响应类、请求类和 HTTP 异常类
from fastapi.responses import StreamingResponse # 从 FastAPI 导入流式响应类
from pydantic import BaseModel # 从 Pydantic 导入数据验证和模型基类
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList # 从 transformers 库导入自回归语言模型、分词器和停止条件相关类
from peft import PeftModel # 导入PEFT库的PeftModel用于加载LoRA权重

# =============================================
# 2. 全局配置
# =============================================

# --- 模型路径或标识符 ---
# 聊天模型路径 (指定本地 Qwen-14B-Chat 模型快照)
chat_model_path = "/root/autodl-tmp/Hugging-Face/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8" # 定义聊天模型的本地路径
# LoRA检查点路径 (指定最佳检查点)
lora_checkpoint_path = "/root/autodl-tmp/save/llama2-lora-2025-04-05-13-22-23/checkpoint-537" # 定义LoRA检查点路径

# =============================================
# 2.5 自定义 StoppingCriteria
# =============================================
# 定义一个在生成特定 token 后停止的自定义停止条件类
class StopOnChatTurnTokensCriteria(StoppingCriteria): # 继承自 StoppingCriteria
    """
    自定义停止条件：当模型生成指定的聊天轮换标记 (<|im_start|> 或 <|im_end|>) 时停止生成。
    """
    def __init__(self, chat_turn_token_ids: List[int], prompt_length: int): # 初始化方法
        """
        初始化停止条件。

        Args:
            chat_turn_token_ids: 需要触发停止的特殊 token ID 列表。
            prompt_length: 输入提示 (prompt) 的 token 长度。
        """
        self.chat_turn_token_ids = chat_turn_token_ids # 存储需要停止的 token ID 列表
        self.prompt_length = prompt_length # 存储输入提示的长度

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool: # 定义调用逻辑
        """
        在每个生成步骤被调用，判断是否应该停止生成。

        Args:
            input_ids: 当前已生成的 token ID 序列 (包含 prompt)。
            scores: 当前生成步骤的 logits 分数 (未使用)。
            **kwargs: 其他可能的关键字参数。

        Returns:
            如果应该停止生成，则返回 True；否则返回 False。
        """
        # 检查当前生成的序列长度是否已超过原始 prompt 长度 (即是否生成了新 token)
        if input_ids.shape[1] > self.prompt_length:
            # 获取最新生成的 token ID (序列中最后一个 token)
            last_token = input_ids[0, -1]
            # 检查最新生成的 token 是否在指定的停止 token 列表中
            if last_token in self.chat_turn_token_ids:
                # 如果是停止 token，打印调试信息
                print(f"检测到特殊聊天 token ID: {last_token} (在 prompt 之后)，触发停止条件。")
                # 返回 True，表示停止生成
                return True
        # 如果未生成新 token 或新 token 不是停止 token，则返回 False，继续生成
        return False

# =============================================
# 3. FastAPI 应用初始化
# =============================================
app = FastAPI(title="LLM API Server", description="提供聊天服务 (Embedding 已移除)") # 初始化 FastAPI 应用实例，设置标题和描述

# =============================================
# 4. 模型加载
# =============================================

# --- 加载聊天模型 (Qwen-14B-Chat) ---
print("开始加载聊天模型...") # 打印加载开始信息
# 加载 Qwen 模型的分词器 (Tokenizer)
chat_tokenizer = AutoTokenizer.from_pretrained( # 使用 AutoTokenizer 加载
    chat_model_path,        # 指定模型路径
    local_files_only=True,  # 强制只使用本地缓存文件，不尝试联网下载
    trust_remote_code=True  # 允许执行模型仓库中的自定义 Python 代码 (Qwen 模型需要)
)
# 加载 Qwen 聊天模型本身
chat_model = AutoModelForCausalLM.from_pretrained( # 使用 AutoModelForCausalLM 加载
    chat_model_path,        # 指定模型路径
    local_files_only=True,  # 强制只使用本地缓存文件
    trust_remote_code=True, # 允许执行模型仓库中的自定义 Python 代码
    device_map="auto",      # 自动将模型层分配到可用的设备 (如 GPU 和 CPU)
    torch_dtype=torch.float16, # 指定模型加载时使用的数据类型 (半精度浮点数，节省显存)
    low_cpu_mem_usage=True  # 尝试优化 CPU 内存使用量 (适用于大模型加载)
)

# # 检查LoRA检查点路径是否存在
# if os.path.exists(lora_checkpoint_path):
#     print(f"找到LoRA检查点: {lora_checkpoint_path}，正在加载...")
#     try:
#         # 加载LoRA权重到基础模型
#         chat_model = PeftModel.from_pretrained(
#             base_model,           # 基础模型
#             lora_checkpoint_path, # LoRA检查点路径
#             torch_dtype=torch.float16, # 保持与基础模型相同的数据类型
#             device_map="auto"     # 自动设备映射
#         )
#         print("LoRA检查点加载成功，使用微调后的模型")
#     except Exception as e:
#         print(f"加载LoRA检查点失败: {e}，将使用原始模型")
#         chat_model = base_model
# else:
#     print(f"未找到LoRA检查点: {lora_checkpoint_path}，使用原始模型")
#     chat_model = base_model

# print("聊天模型加载完成.") # 打印加载完成信息

# =============================================
# 5. Pydantic 模型定义 (请求和响应体结构)
# =============================================

# --- 聊天 API 模型 ---
# 定义单个聊天消息的数据结构
class ChatMessage(BaseModel): # 继承自 Pydantic 的 BaseModel
    """单个聊天消息的结构""" # Docstring 描述
    role: str       # 定义 'role' 字段，类型为字符串 (例如 "system", "user", "assistant")
    content: str    # 定义 'content' 字段，类型为字符串 (消息的具体内容)

# 定义聊天补全请求的数据结构 (模仿 OpenAI API)
class ChatCompletionRequest(BaseModel): # 继承自 BaseModel
    """聊天请求的结构 (仿 OpenAI)""" # Docstring 描述
    model: str                  # 定义 'model' 字段，字符串类型 (请求的模型名称)
    messages: List[ChatMessage] # 定义 'messages' 字段，类型为 ChatMessage 对象的列表
    temperature: Optional[float] = 0.7 # 定义 'temperature' 字段，可选的浮点数，默认 0.7 (控制生成随机性)
    max_tokens: Optional[int] = 1000  # 定义 'max_tokens' 字段，可选整数，默认 1000 (控制最大生成长度)
    stream: Optional[bool] = False    # 定义 'stream' 字段，可选布尔值，默认 False (是否使用流式响应)

# 定义 Token 使用量信息的数据结构
class Usage(BaseModel): # 继承自 BaseModel
    """Token 使用量信息结构""" # Docstring 描述
    prompt_tokens: int          # 定义 'prompt_tokens' 字段，整数类型 (输入提示的 Token 数)
    total_tokens: int           # 定义 'total_tokens' 字段，整数类型 (总 Token 数)
    completion_tokens: Optional[int] = None # 定义 'completion_tokens' 字段，可选整数 (生成内容的 Token 数)

# =============================================
# 7. API 端点定义
# =============================================

# --- 标准 OpenAI 聊天端点 --- #
# 定义处理 POST 请求到 "/v1/chat/completions" 路径的函数
@app.post("/v1/chat/completions", tags=["Chat Completions"]) # FastAPI 装饰器，定义路由和标签
async def openai_chat_completions(request: ChatCompletionRequest): # 定义异步处理函数，参数 'request' 类型为 ChatCompletionRequest
    """处理标准 OpenAI 格式的聊天请求 (使用 Qwen 模型)""" # API 端点的 Docstring 描述
    # 记录接收到的请求消息 (用于调试)
    print(f"接收到 /v1/chat/completions 请求。消息列表:") # 打印接收请求的日志
    for msg in request.messages: # 遍历请求中的消息列表
        print(f"  角色: {msg.role}, 内容: {msg.content}") # 打印每条消息的角色和内容

    # --- 使用 Qwen 的 Tokenizer 和 Chat Template 构建 Prompt ---
    try: # 开始异常处理块
        # 将 Pydantic 模型列表转换为 Qwen chat template 需要的字典列表格式
        messages_for_template = [{"role": m.role, "content": m.content} for m in request.messages] # 列表推导式进行格式转换
        # 使用 tokenizer 的 apply_chat_template 方法构建符合 Qwen 格式的输入字符串
        prompt_str = chat_tokenizer.apply_chat_template( # 调用模板应用方法
            messages_for_template, # 传入转换后的消息列表
            tokenize=False,        # 指示不进行 tokenize，返回字符串
            add_generation_prompt=True # 添加指示模型开始生成回复的特殊标记
        )
        print(f"\n构建的 Qwen Prompt:\n{prompt_str}\n") # 打印构建好的 Prompt 字符串 (用于调试)
    except Exception as e: # 捕获构建 Prompt 过程中可能出现的任何异常
        print(f"错误: 构建 Qwen Prompt 失败: {e}") # 打印错误信息
        raise HTTPException(status_code=500, detail="处理请求时构建 Prompt 失败") # 抛出 HTTP 500 错误

    # --- 准备输入并获取 Prompt 长度 ---
    try: # 开始异常处理块
        # 使用 tokenizer 对构建好的 prompt 字符串进行编码，返回 PyTorch 张量，并包含 attention_mask
        inputs = chat_tokenizer(prompt_str, return_tensors="pt", return_attention_mask=True).to(chat_model.device) # 分词并移动到模型所在设备
        # 获取输入 token ID 的长度 (即 Prompt 的 token 数量)
        prompt_length = inputs["input_ids"].shape[1] # 从 input_ids 张量的形状获取长度
        print(f"Prompt 长度: {prompt_length}") # 打印 Prompt 长度 (用于调试)
    except Exception as e: # 捕获准备输入过程中可能出现的任何异常
        print(f"错误: 准备模型输入或获取 Prompt 长度失败: {e}") # 打印错误信息
        raise HTTPException(status_code=500, detail="准备模型输入时出错") # 抛出 HTTP 500 错误

    # --- 配置生成参数 ---
    # 创建一个字典来存储传递给模型 generate 方法的参数
    generation_config = {
        "max_new_tokens": min(request.max_tokens, 500), # 设置最大生成的新 token 数量，增加上限控制
        "temperature": min(request.temperature, 0.5), # 限制温度参数不超过0.5，降低随机性
        "do_sample": True if request.temperature > 0 else False, # 根据温度决定是否进行采样
        "top_p": 0.8, # 降低 top-p 值，使生成更加集中
        "repetition_penalty": 1.2, # 增加重复惩罚因子，更强力地避免重复
        "eos_token_id": chat_tokenizer.eos_token_id, # 设置序列结束符的 token ID
        "pad_token_id": chat_tokenizer.pad_token_id, # 设置填充符的 token ID
        "no_repeat_ngram_size": 3, # 添加 n-gram 惩罚，避免 n-gram 短语重复
    }
    print(f"使用的生成参数: {generation_config}") # 打印使用的生成参数

    # --- 实例化 StoppingCriteria (传入 Prompt 长度) ---
    # Qwen 模型的特殊聊天轮换 token ID 和常见噪声词
    chat_turn_token_ids = [151644, 151645] # <|im_start|> 和 <|im_end|> 的 ID
    
    # --- 创建自定义停止条件 ---
    stop_criteria = StopOnChatTurnTokensCriteria(
        chat_turn_token_ids=chat_turn_token_ids,
        prompt_length=prompt_length
    )
    stopping_criteria_list = StoppingCriteriaList([stop_criteria])

    # --- 调用模型生成文本 (传入 StoppingCriteria 和 attention_mask) ---
    try: # 开始异常处理块
        with torch.no_grad(): # 在不计算梯度的上下文中执行，节省内存和计算资源
            # 调用模型的 generate 方法生成文本
            outputs = chat_model.generate( # 调用生成方法
                input_ids=inputs["input_ids"], # 传入编码后的输入 token ID
                attention_mask=inputs["attention_mask"], # 传入 attention mask 以忽略填充部分
                **generation_config, # 解包传入生成配置参数字典
                stopping_criteria=stopping_criteria_list # 传入自定义的停止条件列表
            )
    except Exception as e: # 捕获模型生成过程中可能出现的任何异常
        print(f"错误: 模型生成失败: {e}") # 打印错误信息
        raise HTTPException(status_code=500, detail="模型生成文本时出错") # 抛出 HTTP 500 错误

    # --- 解码并提取生成的文本 ---
    try: # 开始异常处理块
        # 使用 tokenizer 解码模型输出的完整 token 序列
        full_generated_text = chat_tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"\n模型完整生成文本 (含 Prompt):\n{full_generated_text}\n")

        # 使用 tokenizer 解码模型输出中仅生成的部分
        generated_text = chat_tokenizer.decode(
            outputs[0][prompt_length:],
            skip_special_tokens=True
        ).strip()
        print(f"\n提取的生成文本 (原始):\n{generated_text}\n")
        
        # --- 添加后处理步骤，清理明显的干扰内容 ---
        # 清理常见的噪声标记和模式
        noisy_patterns = [
            "what", "whatuser", "whatassistant", 
            "user:", "assistant:", "system:", "用户：", "助手：",
            "Human:", "Assistant:", "User:", "Assitant:",
            "根据以上对话提出的问题是", "根据以上对话回答的问题是",
            "请问您还有其他问题需要解答吗", "如果您想了解更详细的信息",
            "如果您有任何其他问题", "请随时告诉我"
        ]
        
        # 处理角色标记和自我对话的清理
        dialogue_markers = [
            "Assistant:", "Human:", "User:", "Assitant:", 
            "assistant:", "user:", "system:", 
            "用户：", "助手：", "系统："
        ]
        
        # 检测是否存在对话标记
        for marker in dialogue_markers:
            if marker in generated_text:
                parts = generated_text.split(marker, 1)
                if len(parts) > 1 and len(parts[0].strip()) > 0:  # 确保分割前有内容
                    # 保留第一个有效回复
                    generated_text = parts[0].strip()
                    print(f"检测到对话标记 '{marker}'，保留前面的内容")
                    break
        
        # 找出噪声模式的位置
        first_noise_pos = float('inf')
        for pattern in noisy_patterns:
            pos = generated_text.lower().find(pattern.lower())  # 不区分大小写查找
            if pos > len(generated_text) // 3:  # 只处理文本前1/3之后的噪声
                if pos < first_noise_pos:
                    first_noise_pos = pos
                    print(f"在位置 {pos} 检测到噪声模式 '{pattern}'")
        
        # 如果找到噪声，截断文本
        if first_noise_pos < float('inf'):
            print(f"在位置 {first_noise_pos} 处截断文本")
            generated_text = generated_text[:first_noise_pos].strip()
        
        # 删除特定角色的自我标识和常见结尾语
        self_identifiers = [
            "我是一名AI语言模型", "我是AI语言模型", "我是一个AI", "作为AI", 
            "作为一个AI", "作为人工智能", "作为一个人工智能",
            "我是一名AI助手", "我是AI助手", "我是一个助手",
            "如果您需要更多信息", "如果您有任何疑问",
            "希望这些信息对您有帮助", "欢迎继续提问"
        ]
        
        # 检测文本中的自我标识，但保护开头的必要自我介绍
        for identifier in self_identifiers:
            pos = generated_text.find(identifier)
            if pos > len(generated_text) // 3:  # 只处理文本前1/3之后的自我标识
                generated_text = generated_text[:pos].strip()
                print(f"检测到自我描述或结尾语，在位置 {pos} 处截断文本")
                break
        
        # 检测重复内容
        min_repeat_length = 15  # 增加最小重复长度阈值
        text_length = len(generated_text)
        
        # 将文本分成更小的片段进行重复检测
        for length in range(min_repeat_length, min(100, text_length // 2)):
            found_repeat = False
            for i in range(text_length - length * 2):
                pattern = generated_text[i:i + length]
                if len(pattern.strip()) < min_repeat_length:
                    continue
                    
                # 在剩余文本中查找重复，但跳过紧邻的文本
                remaining_text = generated_text[i + length + 50:]  # 添加间隔
                if pattern in remaining_text:
                    print(f"检测到重复内容：'{pattern}'")
                    # 截断到第一次出现的位置
                    generated_text = generated_text[:i + length].strip()
                    found_repeat = True
                    break
            if found_repeat:
                break
        
        # 清理多行空白并标准化换行
        generated_text = "\n".join([line.strip() for line in generated_text.splitlines() if line.strip()])
        
        print(f"\n清理后的生成文本:\n{generated_text}\n")
    except Exception as e:
        print(f"错误: 解码或提取回复失败: {e}")
        raise HTTPException(status_code=500, detail="处理模型输出时出错")

    # --- 改进的重复内容检测 ---
    if len(generated_text) > 100:  # 对较长文本执行重复检测
        # 将长文本分割成更小的片段 (30字符)，使检测更精确
        segments = [generated_text[i:i+30] for i in range(0, len(generated_text)-30, 10)]  # 使用更小窗口和重叠步长
        
        for i in range(len(segments)-2):  # 至少需要一定间隔才判定为重复
            segment = segments[i]
            if len(segment.strip()) < 10:  # 忽略太短的片段
                continue
                
            # 在后面的文本中查找当前片段
            latter_text = generated_text[i*10+60:]  # 跳过相邻片段
            if segment in latter_text:
                print(f"警告: 检测到重复内容: '{segment}'，已截断。")
                # 截断到重复内容之前
                cutoff_pos = i*10 + 30
                generated_text = generated_text[:cutoff_pos].strip()
                break

    # --- 处理响应格式 (流式或非流式) ---
    # 检查请求是否要求流式响应
    if request.stream:
        # --- 处理流式响应 (Server-Sent Events, SSE) ---
        # 定义一个异步生成器函数来产生 SSE 数据块
        async def stream_generator(): # 定义异步生成器
            # 生成一个唯一的响应 ID (结合时间戳和随机数)
            response_id = f"chatcmpl-{int(time.time() * 1000)}{torch.randint(100, 999, (1,)).item()}" # 创建唯一 ID
            # 获取当前时间的 Unix 时间戳
            created_time = int(time.time()) # 获取创建时间

            # 1. 发送包含角色信息的第一个 chunk
            # 构建第一个数据块，包含元数据和空的 'delta' 内容，角色为 'assistant'
            first_chunk_data = {
                'id': response_id, # 响应 ID
                'object': 'chat.completion.chunk', # 对象类型
                'created': created_time, # 创建时间戳
                'model': request.model, # 请求的模型名称
                'choices': [{ # 选择列表 (通常只有一个)
                    'delta': {'role': 'assistant', 'content': ''}, # 增量内容，第一个包含角色
                    'index': 0, # 选择索引
                    'finish_reason': None # 结束原因 (尚未结束)
                }]
            }
            # 使用 SSE 格式发送第一个数据块 (data: {json_payload}\n\n)
            yield f"data: {json.dumps(first_chunk_data)}\n\n" # yield 发送数据块

            # 2. 分块发送实际生成的文本内容
            chunk_size = 50 # 定义每个文本块的大小 (字符数)
            # 遍历生成的文本，按 chunk_size 步长迭代
            for i in range(0, len(generated_text), chunk_size):
                # 提取当前块的文本内容
                chunk_content = generated_text[i:min(i + chunk_size, len(generated_text))] # 获取文本片段
                # 构建包含文本增量的数据块
                chunk_data = {
                    'id': response_id, # 响应 ID
                    'object': 'chat.completion.chunk', # 对象类型
                    'created': created_time, # 创建时间戳
                    'model': request.model, # 请求的模型名称
                    'choices': [{ # 选择列表
                        'delta': {'content': chunk_content}, # 增量内容，只包含当前块的文本
                        'index': 0, # 选择索引
                        'finish_reason': None # 结束原因 (尚未结束)
                    }]
                }
                # 使用 SSE 格式发送当前文本块
                yield f"data: {json.dumps(chunk_data)}\n\n" # yield 发送数据块
                # 短暂暂停，模拟生成过程，并防止发送过快
                await asyncio.sleep(0.02) # 异步等待

            # 3. 发送包含结束信号和 token 使用量的最后一个 chunk
            # 计算输入提示的 token 数量
            prompt_tokens = len(inputs["input_ids"][0]) # 获取 prompt token 数
            # 对生成的文本进行编码，计算其 token 数量
            completion_tokens = len(chat_tokenizer.encode(generated_text)) # 获取 completion token 数
            # 计算总 token 数量
            total_tokens = prompt_tokens + completion_tokens # 计算总 token 数

            # 构建最后一个数据块，包含空的 'delta'，结束原因为 'stop'，以及 token 使用量
            finish_chunk_data = {
                'id': response_id, # 响应 ID
                'object': 'chat.completion.chunk', # 对象类型
                'created': created_time, # 创建时间戳
                'model': request.model, # 请求的模型名称
                'choices': [{ # 选择列表
                    'delta': {}, # 结束块的 delta 为空
                    'index': 0, # 选择索引
                    'finish_reason': 'stop' # 标记结束原因为 'stop'
                }],
                'usage': { # 添加 token 使用量信息
                    'prompt_tokens': prompt_tokens, # 输入 token 数
                    'completion_tokens': completion_tokens, # 生成 token 数
                    'total_tokens': total_tokens # 总 token 数
                }
            }
            # 使用 SSE 格式发送结束块
            yield f"data: {json.dumps(finish_chunk_data)}\n\n" # yield 发送数据块

            # 4. 发送 SSE 协议的结束标记
            yield "data: [DONE]\n\n" # 发送 SSE 结束信号

        # 返回一个 StreamingResponse，将异步生成器包装起来，并设置正确的媒体类型
        return StreamingResponse(stream_generator(), media_type="text/event-stream") # 返回流式响应对象

    else:
        # --- 处理非流式响应 ---
        # 计算输入提示的 token 数量
        prompt_tokens = len(inputs["input_ids"][0]) # 获取 prompt token 数
        # 对生成的文本进行编码，计算其 token 数量
        completion_tokens = len(chat_tokenizer.encode(generated_text)) # 获取 completion token 数
        # 计算总 token 数量
        total_tokens = prompt_tokens + completion_tokens # 计算总 token 数

        # 构建完整的 JSON 响应数据结构 (模仿 OpenAI)
        response_data = {
            "id": f"chatcmpl-{int(time.time() * 1000)}{torch.randint(100, 999, (1,)).item()}", # 生成唯一响应 ID
            "object": "chat.completion", # 对象类型
            "created": int(time.time()), # 创建时间戳
            "model": request.model, # 请求的模型名称
            "choices": [ # 选择列表 (包含完整回复)
                {
                    "index": 0, # 选择索引
                    "message": { # 完整的消息对象
                        "role": "assistant", # 角色
                        "content": generated_text # 完整的生成内容
                    },
                    "finish_reason": "stop" # 结束原因
                }
            ],
            "usage": { # Token 使用量信息
                "prompt_tokens": prompt_tokens, # 输入 token 数
                "completion_tokens": completion_tokens, # 生成 token 数
                "total_tokens": total_tokens # 总 token 数
            }
        }
        # 返回一个 FastAPI 的 Response 对象
        return Response( # 创建响应对象
            content=json.dumps(response_data, ensure_ascii=False), # 将响应数据序列化为 JSON 字符串，确保 UTF-8 字符正确处理
            media_type="application/json; charset=utf-8" # 设置响应的媒体类型和字符集
        )

# --- RAGFlow 兼容聊天端点 --- #
# 定义处理 POST 请求到 "/generate/v1/chat/completions" 路径的函数 (兼容 RAGFlow)
@app.post("/generate/v1/chat/completions", tags=["Compatibility"]) # FastAPI 装饰器，定义路由和标签
async def ragflow_chat_completions(request: ChatCompletionRequest): # 定义异步处理函数，参数 'request' 类型为 ChatCompletionRequest
    """RAGFlow 兼容路径，转发到标准聊天端点""" # API 端点的 Docstring 描述
    print("接收到 /generate/v1/chat/completions (RAGFlow 兼容) 请求，转发中...") # 打印兼容请求的日志
    # 直接调用并返回标准聊天端点的处理结果
    return await openai_chat_completions(request) # await 调用标准端点函数

# =============================================
# 8. 服务启动入口
# =============================================
# 检查脚本是否作为主模块运行
if __name__ == "__main__": # Python 标准入口检查
    # 使用 uvicorn 启动 FastAPI 应用
    # host="0.0.0.0" 表示监听所有可用的网络接口，允许外部访问
    # port=8000 指定服务监听的端口号
    import uvicorn # 导入 uvicorn ASGI 服务器
    print("启动 Uvicorn 服务器...") # 打印启动服务器信息
    # 运行 uvicorn 服务器
    uvicorn.run(app, host="0.0.0.0", port=8000) # 传入 FastAPI 应用实例、主机和端口