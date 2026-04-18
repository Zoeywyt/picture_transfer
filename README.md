# FastAPI LLM API 服务器

## 描述

这是一个使用 FastAPI 构建的 Python Web 服务器，旨在提供与 OpenAI API 兼容的接口，用于本地部署的大语言模型 (LLM)。它目前支持：

*   **聊天 (Chat Completions):** 使用本地部署的 `meta-llama/Llama-2-13b-chat-hf` 模型。
*   **文本嵌入 (Embeddings):** 使用本地部署的 `nomic-ai/nomic-embed-text-v1` 模型。

服务器还包含特定的兼容性端点，以方便与 RAGFlow 等应用集成。

## 特性

*   提供符合 OpenAI 规范的 `/v1/chat/completions` 和 `/v1/embeddings` 端点。
*   使用本地缓存的 Hugging Face 模型，无需每次请求都联网。
*   支持 Llama 2 聊天模型的特定提示格式。
*   支持 Nomic Embedding 模型及 Mean Pooling。
*   支持流式 (stream) 和非流式聊天响应。
*   包含 RAGFlow 兼容的端点 (`/generate/v1/...` 和 `/embed`, `/v1/embed`)。
*   自动检测并使用 GPU (如果 CUDA 可用)。

## 先决条件

*   Python 3.8+
*   PyTorch (建议安装支持 CUDA 的版本以利用 GPU)
*   Transformers 库
*   FastAPI
*   Uvicorn (用于运行 FastAPI 应用)
*   NumPy
*   (可选) `ninja` (如果需要重新编译某些依赖项)
*   (可选) `megablocks` (如果希望 Nomic Embedding 模型获得最佳性能并消除相关警告，但安装可能涉及编译)

建议在一个虚拟环境（如 Conda 或 venv）中安装这些依赖。

## 模型设置

1.  **下载模型:** 您需要预先将 `meta-llama/Llama-2-13b-chat-hf` 和 `nomic-ai/nomic-embed-text-v1` 模型下载到本地。脚本默认期望模型位于 `/root/autodl-tmp/Hugging-Face/hub/` 目录下，并遵循 Hugging Face Hub 的缓存结构（例如，包含 `snapshots/<commit_hash>/...` 子目录）。
2.  **配置路径:**
    *   **聊天模型:** 脚本中的 `chat_model_path` 变量必须指向 Llama 2 模型快照 (snapshot) 的确切本地路径。
    *   **Embedding 模型:** 脚本使用 `embedding_model_identifier = "nomic-ai/nomic-embed-text-v1"`。`transformers` 库会尝试在 Hugging Face 缓存目录中查找此标识符对应的文件。
3.  **Hugging Face 缓存:**
    *   如果您的模型缓存不在默认位置 (`~/.cache/huggingface/hub`)，您需要通过设置 `HF_HOME` 环境变量来告知 `transformers` 库正确的位置。脚本中注释掉了 `os.environ['HF_HOME'] = '/root/autodl-tmp/Hugging-Face'` 这一行，如果需要可以取消注释，或者最好在系统环境 (如 `.bashrc`) 中设置它。
    *   `.bashrc` 示例:
        ```bash
        export HF_HOME=/root/autodl-tmp/Hugging-Face
        # 如果需要通过镜像下载模型或代码文件 (如下载 nomic 的远程代码):
        export HF_ENDPOINT=https://hf-mirror.com
        ```
        修改 `.bashrc` 后需要运行 `source ~/.bashrc` 或重新登录 shell。

## 安装依赖

在一个配置好 Python 的环境中，安装主要依赖：

```bash
pip install fastapi uvicorn transformers torch numpy "pydantic>=2.0"
```
*(请根据您的系统和是否使用 GPU 安装合适的 PyTorch 版本)*

## 运行服务器

在包含 `main.py` 的 `App` 目录下，运行：

```bash
python main.py
```

服务器将启动，并默认监听在 `0.0.0.0:8000`。您将在终端看到模型加载信息和 Uvicorn 的启动日志。

## 项目演示
<a href="https://github.com/Zoeywyt/picture_transfer/raw/master/App/display.mp4">
  <img src="https://github.com/Zoeywyt/picture_transfer/raw/master/App/demo.png" alt="视频封面" width="600"/>
</a>

## API 端点

### 聊天

*   **`POST /v1/chat/completions`**
    *   **描述:** 标准 OpenAI 聊天端点。
    *   **请求体:** `ChatCompletionRequest` (包含 `model`, `messages`, `temperature`, `max_tokens`, `stream`)。
    *   **响应:** 符合 OpenAI 规范的聊天完成响应 (流式或非流式)。
    *   **模型:** 使用 `chat_model_path` 指定的 Llama 2 模型。
    *   **提示格式:** 内部会将 `messages` 转换为 Llama 2 的 `[INST]` 格式。

*   **`POST /generate/v1/chat/completions`**
    *   **描述:** RAGFlow 兼容端点，内部直接调用 `/v1/chat/completions`。

### Embedding

*   **`POST /v1/embeddings`**
    *   **描述:** 标准 OpenAI Embedding 端点。
    *   **请求体:** `EmbeddingRequest` (包含 `input`, `model`)。
    *   **响应:** 符合 OpenAI 规范的 Embedding 响应。
    *   **模型:** 使用 `embedding_model_identifier` 指定的 Nomic 模型。

*   **`POST /generate/v1/embeddings`**
    *   **描述:** RAGFlow 兼容端点，内部直接调用 `/v1/embeddings`。

*   **`POST /embed`** 和 **`POST /v1/embed`**
    *   **描述:** 主要用于兼容 RAGFlow 添加模型时的特殊测试请求。
    *   **请求体:**
        *   标准格式: `{"input": ..., "model": ...}` (同 `/v1/embeddings`)
        *   RAGFlow 测试格式: `{"inputs": "..."}`
    *   **响应:**
        *   对于标准格式，返回标准 Embedding 响应。
        *   对于 RAGFlow 测试格式，计算测试字符串的 embedding 并返回标准 Embedding 响应结构。

## 注意事项

*   **`local_files_only=True`:** 聊天模型加载时默认强制使用本地文件。Embedding 模型加载时则允许网络访问（通过 `HF_ENDPOINT`）以下载必要的远程代码。如果确信所有文件（包括代码）已在本地缓存，可以将 Embedding 模型加载也改为 `local_files_only=True`。
*   **`trust_remote_code=True`:** 运行这两个模型都需要信任并执行 Hugging Face Hub 上模型仓库中的自定义 Python 代码。请确保您信任这些模型的来源。
*   **`megablocks` 警告:** 您可能会在启动时看到关于 `megablocks` 或 `Grouped GEMM not available` 的警告。这通常不影响功能，但可能会影响 Nomic Embedding 的性能。可以忽略，或尝试安装/重新编译 `megablocks`（可能涉及复杂的编译依赖）。
*   **RAGFlow 兼容性:** `/embed` 和 `/v1/embed` 端点的逻辑是为了通过 RAGFlow 添加模型时的测试调用而特别调整的。如果 RAGFlow 的行为发生变化，可能需要再次调整。 
