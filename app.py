import os
import json
import requests
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import gradio as gr
from fastapi.responses import HTMLResponse
import uvicorn
import threading
import sys
import time
from transformers import StoppingCriteria, StoppingCriteriaList

# 设置递归深度限制
sys.setrecursionlimit(10000)

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the LLM API Server!"}


# Model paths
chat_model_path = "/root/autodl-tmp/Hugging-Face/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
lora_checkpoint_path = "/root/autodl-tmp/save/llama2-lora-2025-04-05-13-22-23/checkpoint-537"

# Load tokenizer and model
chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_path, local_files_only=True)
chat_model = AutoModelForCausalLM.from_pretrained(
    chat_model_path,
    local_files_only=True,
    torch_dtype=torch.float16
)

# Ensure pad_token is set to avoid recursion error
if chat_tokenizer.pad_token is None:
    # Try using a different special token
    chat_tokenizer.pad_token = "<PAD>"

if os.path.exists(lora_checkpoint_path):
    try:
        chat_model = PeftModel.from_pretrained(
            chat_model,
            lora_checkpoint_path,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Failed to load LoRA checkpoint: {e}")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.5
    max_tokens: int = 2000
    stream: bool = False


@app.post("/v1/chat/completions", tags=["Chat Completions"])
async def openai_chat_completions(request: ChatCompletionRequest):
    """处理标准 OpenAI 格式的聊天请求 (使用 Qwen 模型)"""
    # 记录接收到的请求消息 (用于调试)
    print(f"接收到 /v1/chat/completions 请求。消息列表:")
    for msg in request.messages:
        print(f"  角色: {msg['role']}, 内容: {msg['content']}")

    # --- 使用 Qwen 的 Tokenizer 和 Chat Template 构建 Prompt ---
    try:
        # 将 Pydantic 模型列表转换为 Qwen chat template 需要的字典列表格式
        messages_for_template = [{"role": m["role"], "content": m["content"]} for m in request.messages]
        # 使用 tokenizer 的 apply_chat_template 方法构建符合 Qwen 格式的输入字符串
        prompt_str = chat_tokenizer.apply_chat_template(
            messages_for_template,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"\n构建的 Qwen Prompt:\n{prompt_str}\n")
    except Exception as e:
        print(f"错误: 构建 Qwen Prompt 失败: {e}")
        raise HTTPException(status_code=500, detail="处理请求时构建 Prompt 失败")

    # --- 准备输入并获取 Prompt 长度 ---
    try:
        # 使用 tokenizer 对构建好的 prompt 字符串进行编码，返回 PyTorch 张量，并包含 attention_mask
        inputs = chat_tokenizer(prompt_str, return_tensors="pt", return_attention_mask=True).to(chat_model.device)
        # 获取输入 token ID 的长度 (即 Prompt 的 token 数量)
        prompt_length = inputs["input_ids"].shape[1]
        print(f"Prompt 长度: {prompt_length}")
    except Exception as e:
        print(f"错误: 准备模型输入或获取 Prompt 长度失败: {e}")
        raise HTTPException(status_code=500, detail="准备模型输入时出错")

    # --- 配置生成参数 ---
    # 创建一个字典来存储传递给模型 generate 方法的参数
    generation_config = {
        "max_new_tokens": min(request.max_tokens, 2000),  # 增大到 1000，可根据情况调整
        "temperature": min(request.temperature, 0.5),
        "do_sample": True if request.temperature > 0 else False,
        "top_p": 0.8,
        "repetition_penalty": 1.2,
        "eos_token_id": chat_tokenizer.eos_token_id,
        "pad_token_id": chat_tokenizer.pad_token_id,
        "no_repeat_ngram_size": 3,
    }
    print(f"使用的生成参数: {generation_config}")

    # --- 实例化 StoppingCriteria (传入 Prompt 长度) ---
    # Qwen 模型的特殊聊天轮换 token ID 和常见噪声词
    chat_turn_token_ids = [151644, 151645]  # <|im_start|> 和 <|im_end|> 的 ID

    class StopOnChatTurnTokensCriteria(StoppingCriteria):
        def __init__(self, chat_turn_token_ids, prompt_length):
            self.chat_turn_token_ids = chat_turn_token_ids
            self.prompt_length = prompt_length

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            generated_tokens = input_ids[0, self.prompt_length:]
            if len(generated_tokens) == 0:
                return False
            last_token = generated_tokens[-1].item()
            return last_token in self.chat_turn_token_ids

    # --- 创建自定义停止条件 ---
    stop_criteria = StopOnChatTurnTokensCriteria(
        chat_turn_token_ids=chat_turn_token_ids,
        prompt_length=prompt_length
    )
    stopping_criteria_list = StoppingCriteriaList([stop_criteria])

    # --- 调用模型生成文本 (传入 StoppingCriteria 和 attention_mask) ---
    try:
        with torch.no_grad():
            # 调用模型的 generate 方法生成文本
            outputs = chat_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_config,
                stopping_criteria=stopping_criteria_list
            )
    except Exception as e:
        print(f"错误: 模型生成失败: {e}")
        raise HTTPException(status_code=500, detail="模型生成文本时出错")

    # --- 解码并提取生成的文本 ---
    try:
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
                if len(parts) > 1 and len(parts[0].strip()) > 0:
                    # 保留第一个有效回复
                    generated_text = parts[0].strip()
                    print(f"检测到对话标记 '{marker}'，保留前面的内容")
                    break

        # 找出噪声模式的位置
        first_noise_pos = float('inf')
        for pattern in noisy_patterns:
            pos = generated_text.lower().find(pattern.lower())
            if pos > len(generated_text) // 3:
                if pos < first_noise_pos:
                    first_noise_pos = pos
                    print(f"在位置 {pos} 检测到噪声模式 '{pattern}'")

        # 如果找到噪声，截断文本
        if first_noise_pos < float('inf'):
            print(f"在位置 {first_noise_pos} 处截断文本")
            # 适当放宽截断条件，比如多保留一些字符
            generated_text = generated_text[:first_noise_pos + 10].strip()

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
            if pos > len(generated_text) // 3:
                # 适当放宽截断条件，比如多保留一些字符
                generated_text = generated_text[:pos + 10].strip()
                print(f"检测到自我描述或结尾语，在位置 {pos} 处截断文本")
                break

        # 检测重复内容
        min_repeat_length = 15
        text_length = len(generated_text)

        # 将文本分成更小的片段进行重复检测
        for length in range(min_repeat_length, min(100, text_length // 2)):
            found_repeat = False
            for i in range(text_length - length * 2):
                pattern = generated_text[i:i + length]
                if len(pattern.strip()) < min_repeat_length:
                    continue

                # 在剩余文本中查找重复，但跳过紧邻的文本
                remaining_text = generated_text[i + length + 100:]  # 增大间隔
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
    if len(generated_text) > 100:
        # 将长文本分割成更小的片段 (30字符)，使检测更精确
        segments = [generated_text[i:i + 30] for i in range(0, len(generated_text) - 30, 10)]

        for i in range(len(segments) - 2):
            segment = segments[i]
            if len(segment.strip()) < 10:
                continue

            # 在后面的文本中查找当前片段
            latter_text = generated_text[i * 10 + 60:]
            if segment in latter_text:
                print(f"警告: 检测到重复内容: '{segment}'，已截断。")
                # 截断到重复内容之前
                cutoff_pos = i * 10 + 30
                generated_text = generated_text[:cutoff_pos].strip()
                break

    # --- 处理响应格式 (流式或非流式) ---
    # 检查请求是否要求流式响应
    if request.stream:
        # --- 处理流式响应 (Server-Sent Events, SSE) ---
        # 定义一个异步生成器函数来产生 SSE 数据块
        async def stream_generator():
            try:
                # 生成一个唯一的响应 ID (结合时间戳和随机数)
                response_id = f"chatcmpl-{int(time.time() * 1000)}{torch.randint(100, 999, (1,)).item()}"
                # 获取当前时间的 Unix 时间戳
                created_time = int(time.time())

                # 1. 发送包含角色信息的第一个 chunk
                # 构建第一个数据块，包含元数据和空的 'delta' 内容，角色为 'assistant'
                first_chunk_data = {
                    'id': response_id,
                    'object': 'chat.completion.chunk',
                    'created': created_time,
                    'model': request.model,
                    'choices': [{
                        'delta': {'role': 'assistant', 'content': ''},
                        'index': 0,
                        'finish_reason': None
                    }]
                }
                # 使用 SSE 格式发送第一个数据块 (data: {json_payload}\n\n)
                yield f"data: {json.dumps(first_chunk_data)}\n\n"

                # 2. 分块发送实际生成的文本内容
                chunk_size = 50
                # 遍历生成的文本，按 chunk_size 步长迭代
                for i in range(0, len(generated_text), chunk_size):
                    # 提取当前块的文本内容
                    chunk_content = generated_text[i:min(i + chunk_size, len(generated_text))]
                    # 构建包含文本增量的数据块
                    chunk_data = {
                        'id': response_id,
                        'object': 'chat.completion.chunk',
                        'created': created_time,
                        'model': request.model,
                        'choices': [{
                            'delta': {'content': chunk_content},
                            'index': 0,
                            'finish_reason': None
                        }]
                    }
                    # 使用 SSE 格式发送当前文本块
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    # 短暂暂停，模拟生成过程，并防止发送过快
                    await asyncio.sleep(0.02)

                # 3. 发送包含结束信号和 token 使用量的最后一个 chunk
                # 计算输入提示的 token 数量
                prompt_tokens = len(inputs["input_ids"][0])
                # 对生成的文本进行编码，计算其 token 数量
                completion_tokens = len(chat_tokenizer.encode(generated_text))
                # 计算总 token 数量
                total_tokens = prompt_tokens + completion_tokens

                # 构建最后一个数据块，包含空的 'delta'，结束原因为 'stop'，以及 token 使用量
                finish_chunk_data = {
                    'id': response_id,
                    'object': 'chat.completion.chunk',
                    'created': created_time,
                    'model': request.model,
                    'choices': [{
                        'delta': {},
                        'index': 0,
                        'finish_reason':'stop'
                    }],
                    'usage': {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': total_tokens
                    }
                }
                # 使用 SSE 格式发送结束块
                yield f"data: {json.dumps(finish_chunk_data)}\n\n"

                # 4. 发送 SSE 协议的结束标记
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"流式响应生成过程中出错: {e}")

        import asyncio
        # 返回一个 StreamingResponse，将异步生成器包装起来，并设置正确的媒体类型
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    else:
        # --- 处理非流式响应 ---
        # 计算输入提示的 token 数量
        prompt_tokens = len(inputs["input_ids"][0])
        # 对生成的文本进行编码，计算其 token 数量
        completion_tokens = len(chat_tokenizer.encode(generated_text))
        # 计算总 token 数量
        total_tokens = prompt_tokens + completion_tokens

        # 构建完整的 JSON 响应数据结构 (模仿 OpenAI)
        response_data = {
            "id": f"chatcmpl-{int(time.time() * 1000)}{torch.randint(100, 999, (1,)).item()}",  # 生成唯一响应 ID
            "object": "chat.completion",  # 对象类型
            "created": int(time.time()),  # 创建时间戳
            "model": request.model,  # 请求的模型名称
            "choices": [  # 选择列表 (包含完整回复)
                {
                    "index": 0,  # 选择索引
                    "message": {  # 完整的消息对象
                        "role": "assistant",  # 角色
                        "content": generated_text  # 完整的生成内容
                    },
                    "finish_reason": "stop"  # 结束原因
                }
            ],
            "usage": {  # Token 使用量信息
                "prompt_tokens": prompt_tokens,  # 输入 token 数
                "completion_tokens": completion_tokens,  # 生成 token 数
                "total_tokens": total_tokens  # 总 token 数
            }
        }
        # 返回一个 FastAPI 的 Response 对象
        return Response(  # 创建响应对象
            content=json.dumps(response_data, ensure_ascii=False),  # 将响应数据序列化为 JSON 字符串，确保 UTF-8 字符正确处理
            media_type="application/json; charset=utf-8"  # 设置响应的媒体类型和字符集
        )


def load_painting_data():
    with open('/root/autodl-tmp/App/json_data/40.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def load_another_painting_data():
    # 请替换为实际的另一个 JSON 文件路径
    with open('/root/autodl-tmp/App/json_data/artworks.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def show_museum_paintings():
    gallery_items = []
    titles = []
    image_folder = '/root/autodl-tmp/App/pictures/图图/奇趣'
    paintings_data_path = '/root/autodl-tmp/App/json_data/奇趣美术馆（按第一到第四季顺序）.json'

    with open(paintings_data_path, 'r', encoding='utf-8') as file:
        paintings_data = json.load(file)

    for index, painting in enumerate(paintings_data, start=1):
        for title, description in painting.items():
            image_file = os.path.join(image_folder, f"图片{index}.png")
            if os.path.exists(image_file):
                gallery_items.append((image_file, f"{title}: {description}"))
                titles.append(title)
            else:
                gallery_items.append((None, f"{title}: {description}"))
                titles.append(title)

    return gallery_items, titles


def show_another_museum_paintings():
    gallery_items = []
    titles = []
    # 请替换为实际的另一个图片文件夹路径
    image_folder = '/root/autodl-tmp/App/pictures/图图/165'
    # 请替换为实际的另一个 JSON 文件路径
    paintings_data_path = '/root/autodl-tmp/App/json_data/165画作主题整合.json'

    with open(paintings_data_path, 'r', encoding='utf-8') as file:
        paintings_data = json.load(file)

    for index, painting in enumerate(paintings_data, start=1):
        for title, description in painting.items():
            image_file = os.path.join(image_folder, f"图片{index}.png")
            if os.path.exists(image_file):
                gallery_items.append((image_file, f"{title}: {description}"))
                titles.append(title)
            else:
                gallery_items.append((None, f"{title}: {description}"))
                titles.append(title)

    return gallery_items, titles


def chat_with_model(prompt, painting_info):
    url = "http://127.0.0.1:8000/v1/chat/completions"  # FastAPI API 的 URL
    headers = {"Content-Type": "application/json"}
    combined_prompt = f"{prompt}\n{painting_info}"
    payload = {
        "model": "Qwen-14B-Chat",
        "messages": [{"role": "user", "content": combined_prompt}],
        "temperature": 0.5,
        "max_tokens": 2000
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # 对错误响应抛出异常
        data = response.json()
        if 'choices' in data and data['choices']:
            return data['choices'][0]['message']['content']
        else:
            return "返回的响应格式不正确。"
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return "与聊天模型通信时出错。"
    except (KeyError, IndexError) as e:
        print(f"解析响应时出错: {e}")
        return "解析聊天模型响应时出错。"


def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 艺术馆")
        gallery_choice = gr.Radio(choices=["美术馆", "各地奇画"], label="选择相册", value="美术馆")

        gallery = gr.Gallery(label="画作展览", columns=5, height="auto")
        painting_title = gr.Textbox(label="画作标题：", interactive=False)
        painting_info = gr.Textbox(label="画作信息：", interactive=False)
        chat_input = gr.Textbox(label="输入聊天内容：")
        chat_output = gr.Textbox(label="聊天回复：", interactive=False)

        def on_gallery_choice_change(choice):
            if choice == "美术馆":
                gallery_items, all_titles = show_museum_paintings()
                painting_data = load_painting_data()
            else:
                gallery_items, all_titles = show_another_museum_paintings()
                painting_data = load_another_painting_data()
            return gallery_items, "", ""

        def on_select(evt: gr.SelectData, choice):
            if choice == "美术馆":
                gallery_items, all_titles = show_museum_paintings()
                painting_data = load_painting_data()
            else:
                gallery_items, all_titles = show_another_museum_paintings()
                painting_data = load_another_painting_data()
            index = evt.index
            if index < len(all_titles):
                title = all_titles[index]
                painting_detail = painting_data.get(title, "No details available.")
                chat_input.interactive = True
                return title, painting_detail
            return "", "No details available."

        gallery_choice.change(on_gallery_choice_change, inputs=gallery_choice, outputs=[gallery, painting_title, painting_info])
        gallery.select(on_select, inputs=[gallery_choice], outputs=[painting_title, painting_info])
        chat_input.submit(fn=chat_with_model, inputs=[chat_input, painting_info], outputs=chat_output)

    return demo


gradio_interface = create_gradio_interface()


@app.get("/gradio", response_class=HTMLResponse)
async def gradio_ui():
    return gradio_interface.launch(prevent_thread_lock=True)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    