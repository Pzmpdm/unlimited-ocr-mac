# Unlimited OCR · Mac (Apple Silicon) 适配版

> **[English](#english)** | **中文**

---

## 中文

在 **Mac Apple Silicon (M1 / M2 / M3 / M4)** 上本地运行 [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)（6.7B VLM）——带 grounding 的 OCR，直接输出 `<|det|><类型> [bbox] <|/det|><文字>` 格式，一次推理同时识别标题、正文、表格、公式与图片区域并给出像素级定位。

**官方只支持 NVIDIA CUDA**。本仓库是社区维护的 Mac 适配版，提供两条推理路径：

| 路径 | 说明 |
|---|---|
| 🚀 **MLX 后端（默认）** | Block float MX FP8 量化模型（~3.6 GB）+ mlx-vlm 流式生成，~205 tok/s、峰值内存 ~5 GB |
| 🧩 **PyTorch/MPS 后端** | 官方 bf16 模型（~6.7 GB）+ 3 个 Mac 补丁，CLI 与备用路径 |

### ✨ 功能亮点

| 功能 | 说明 |
|---|---|
| 🔍 **本地 OCR** | 6.7B VLM 完全本地运行，无需联网，隐私安全 |
| 🖥️ **Web UI** | 浏览器直操作：上传 → 扫描 → 编辑 → 翻译 → 导出 |
| ⚡ **实时流式** | SSE 逐行推送识别结果，边扫边看；支持暂停 / 继续 / 停止 / 自动跟随 |
| 📄 **PDF 原生文本层提取** | 数字版 PDF（带文本层）直接走原生布局提取，表格 / 公式 / 图片结构还原，比 VLM 更快更准；扫描件才调用 VLM |
| ✏️ **在线编辑** | 识别结果可直接点击修改，实时保存，并反映到导出文件 |
| 🌐 **双语翻译** | 支持中/英/日/韩/法/德等 10+ 语言互译，原文+译文紧贴对照 |
| 📥 **双格式导出** | Markdown + 图片 (ZIP)（默认）与 Word（仅原文 / 仅译文 / 双语对照） |
| 🤖 **MCP 服务** | 一键接入 Claude Code / Codex / 其他 AI 工具 |

### 实测性能 (M4 Pro 48GB)

| 任务 | 耗时 / 数据 |
|---|---|
| MLX MXFP8 解码速度 | ~205 tok/s（FP16 基线 ~146 tok/s） |
| MLX 峰值内存 / 磁盘 | ~5.0 GB / ~3.6 GB |
| PyTorch/MPS 模型加载(从 SSD 读 6.7GB) | ~3s |
| 简单图 (768×1024) | ~9s |
| 复杂文档 (1024×1024, 13 行) | ~19s |

### 🚀 安装

#### 一键安装（推荐）

```bash
git clone https://github.com/Pzmpdm/unlimited-ocr-mac.git
cd unlimited-ocr-mac
bash install.sh            # 一键安装 MLX 后端（默认，推荐）
bash install.sh --pytorch  # 如需 PyTorch/MPS 后端
bash install.sh --start    # 安装完成后自动启动 Web UI
```

安装完成后启动：

```bash
bash web/scripts/start_web.sh   # 打开 http://localhost:8800
```

> 💡 一键脚本会自动创建虚拟环境、安装依赖、下载模型（MXFP8 ~3.6 GB + 原始 tokenizer），已存在的内容自动跳过，可放心重复执行。

#### 路径 A：MLX 后端（手动，供参考）

```bash
git clone https://github.com/Pzmpdm/unlimited-ocr-mac.git
cd unlimited-ocr-mac

# 1. 下载 MLX 量化模型 (MXFP8, ~3.6 GB) 与原始 tokenizer
#    模型: https://huggingface.co/sahilchachra/unlimited-ocr-mxfp8-mlx
#    放到仓库外一级目录，例如 ../Unlimited-OCR-MLX（模型）和 ../Unlimited-OCR-model（原始 bf16 权重，仅需 tokenizer.json）
#    可用 OCR_MLX_MODEL_DIR 指定模型目录

# 2. 创建虚拟环境并安装依赖
python3.13 -m venv .venv-mlx
source .venv-mlx/bin/activate
pip install mlx mlx-vlm transformers \
    fastapi sse-starlette python-multipart uvicorn PyMuPDF python-docx

# 3. 启动 Web UI
bash web/scripts/start_web.sh        # 默认端口 8800
# 打开 http://localhost:8800
```

#### 路径 B：PyTorch/MPS 后端（CLI / 备用）

```bash
git clone https://github.com/Pzmpdm/unlimited-ocr-mac.git
cd unlimited-ocr-mac
bash scripts/install.sh          # 安装 OCR 模型 (~5-10 分钟, 下载 6.7GB, 自动打 Mac 补丁)
bash web/scripts/install_web.sh  # 安装 Web UI 依赖 (~1 分钟)
```

模型默认装到 `./model_dir`，缓存走 `~/.cache/huggingface/hub/`。装外置 SSD：

```bash
HF_HUB_CACHE="/Volumes/外置盘/Model/hub" bash scripts/install.sh
```

### 一键运行

**Web UI（推荐）**：

```bash
bash web/scripts/start_web.sh
# 打开 http://localhost:8800
# 默认 MLX 后端；切 PyTorch: OCR_BACKEND=pytorch bash web/scripts/start_web.sh
```

**命令行（PyTorch/MPS 后端）**：

```bash
# 单图
bash scripts/run.sh /path/to/your.png

# 多图批处理
bash scripts/run.sh --image_dir /path/to/images/ --output_dir ./out

# 高级参数
bash scripts/run.sh /path/to/image.png \
    --base_size 1024 --image_size 640 --max_length 8192 \
    --no_repeat_ngram_size 35 --ngram_window 128
```

### 🖥️ Web UI 使用指南

#### 基本流程

```
上传 PDF/图片 → 点击「开始识别」→ 实时查看识别结果 → 编辑 → 翻译 → 导出
```

#### 界面布局

| 区域 | 功能 |
|---|---|
| 左侧边栏 | 打开文档、页面缩略图列表、本机处理标识 |
| 左侧面板 | 原始文档预览（PDF 自动按页切换，缩放 / 适应） |
| 右侧面板 | 识别结果：预览（Markdown 渲染）/ 编辑 双视图切换 |
| 顶部工具栏 | 模型状态 / 主题切换 / 上传 / 扫描 / 导出 / 自动跟随 |
| 底部进度条 | 暂停 / 继续 / 停止 / 计时 |

#### 翻译功能

1. 在顶部下拉框选择目标语言（如 `→ English`）
2. 点击 🌐 翻译 按钮
3. 译文以紫色缩进样式紧跟在原文下方
4. 再次点击可取消翻译

**翻译 API 配置**（编辑 `web/.env`）:

```bash
# 支持任何 OpenAI 兼容的 API
TRANSLATE_API_BASE=https://api.openai.com/v1    # OpenAI
# TRANSLATE_API_BASE=https://api.deepseek.com/v1 # DeepSeek
# TRANSLATE_API_BASE=http://localhost:11434/v1    # 本地 Ollama

TRANSLATE_API_KEY=sk-your-api-key-here
TRANSLATE_MODEL=gpt-4o
```

> 💡 **提示**: 翻译是**可选功能**。不配置 API Key 仍可正常使用 OCR 识别和导出。

#### 导出

| 方式 | 说明 |
|---|---|
| Markdown + 图片 (ZIP)（默认） | `*.md` + `images/`（自动裁剪 OCR 定位到的图片区域） |
| Word (.docx) | 仅原文 / 仅译文 / 双语对照三种模式 |

### 作为 MCP 服务（给 Claude Code / Codex / 其他 AI 工具调用）

```bash
bash scripts/mcp.sh    # 启动 stdio MCP server
```

注册到 Claude Code（`~/.claude.json` 或项目 `.mcp.json`）:
```json
{
  "mcpServers": {
    "unlimited-ocr": {
      "command": "python3",
      "args": ["/path/to/unlimited-ocr-mac/mcp_server/server.py"]
    }
  }
}
```

注册到 OpenAI Codex CLI（`~/.codex/config.toml`）:
```toml
[mcp_servers.unlimited-ocr]
command = "python3"
args = ["/abs/path/to/unlimited-ocr-mac/mcp_server/server.py"]
```

启动后,你的 AI 工具就能调用:
- `ocr_image(path)` — 单张图 OCR，返回 JSON detections
- `ocr_directory(path)` — 批处理目录，保存 .md
- `model_status()` — 查看模型加载状态

### 🏗️ 项目结构

```
unlimited-ocr-mac/
├── patches/              # Mac 适配补丁（覆盖到下载的 HF 模型）
│   ├── modeling_unlimitedocr.py   # .cuda() → .to('mps') + 禁用 autocast
│   ├── modeling_deepseekv2.py     # DeepSeek V2 模型适配
│   └── ...
├── scripts/              # CLI 安装/运行脚本
│   ├── install.sh               # 一键安装 OCR 模型（PyTorch 路径）
│   ├── run.sh                   # 一键运行 OCR（PyTorch/MPS）
│   └── mcp.sh                   # 启动 MCP 服务
├── mcp_server/           # MCP Server（给 AI 工具调用）
│   └── server.py
├── web/                  # 🌐 Web UI（默认 MLX 后端）
│   ├── server.py                # FastAPI 主服务（SSE 实时流）
│   ├── ocr_engine_mlx.py        # MLX 推理引擎（mlx-vlm 流式生成）
│   ├── ocr_engine.py            # PyTorch/MPS 推理引擎（懒加载）
│   ├── ocr_parser.py            # OCR 输出解析 → 结构化 + HTML/Markdown
│   ├── pdf_converter.py         # PDF 渲染 + 原生文本层/表格/公式/图片提取
│   ├── translator.py            # 翻译引擎（OpenAI 兼容 API）
│   ├── docx_exporter.py         # Word 文档导出
│   ├── config.py                # 配置管理
│   ├── public/                  # 前端静态文件
│   │   ├── index.html
│   │   ├── style.css
│   │   └── app.js
│   ├── scripts/
│   │   ├── install_web.sh       # Web UI 一键安装
│   │   └── start_web.sh         # Web UI 一键启动（MLX 默认）
│   ├── .env.example             # 翻译 API 配置模板
│   └── requirements.txt         # Python 依赖
├── install.sh            # 一键安装脚本（MLX 默认 / --pytorch）
├── run_mac.py            # Mac 适配核心（应用 patch + 推理）
└── README.md
```

### 两条 Mac 适配方案

#### 方案一：MLX 量化（默认，推荐）

- 模型：`sahilchachra/unlimited-ocr-mxfp8-mlx`（Block float MX FP8，9.19 有效 bit/权重，~3.6 GB）
- 引擎：mlx-vlm `stream_generate` 流式生成（`.venv-mlx`）
- 解码用百度原始 `tokenizer.json`，保证空格/换行还原准确
- 内置 MLX 版滑窗 n-gram 去重（ngram=35, window=128），等效官方 repetition blocker
- 实时统计 tokens/s、峰值内存，显示在 Web 界面

#### 方案二：PyTorch/MPS 三个补丁（CLI / 备用）

模型官方代码假设 NVIDIA GPU，在 Mac 上要改三处：

1. **`.cuda()` 硬编码 → `.to('mps')`**（`modeling_unlimitedocr.py` 里 13+ 处）
   - `run_mac.py::apply_patches()` 自动改写，生成 `model_dir_patched/`
2. **禁用 `torch.autocast("cuda", ...)`** ⭐ **最关键**
   - PyTorch MPS backend 对 `autocast(device_type="mps", dtype=bfloat16)` 有 bug，会让 MLA+MoE 算子精度漂移
   - 去掉 autocast 后模型用自然 bf16 推理；否则 5-10 token 后 logits 塌缩成 `168168168...` 死循环
3. **`SlidingWindowNoRepeatNgramProcessor` 继承 `LogitsProcessor`**
   - 原版是普通类，transformers 4.57 不会调用它；加继承让 ngram 真正生效

这些 patch 都在 `patches/` 目录里，`install.sh` 自动覆盖到下载的 HF 模型。

### 🔧 系统要求

#### OCR 模型（核心）

| 项目 | 要求 |
|---|---|
| 操作系统 | macOS 14+ (Sonoma) on Apple Silicon (M1/M2/M3/M4) |
| Python | 3.12 或 3.13 (推荐 3.13) |
| 统一内存 | **≥ 24GB**（推荐 32GB+；48GB 实测无压力） |
| 磁盘 | MLX 版约 4GB；PyTorch 版约 8GB（模型权重） |

#### Web UI（额外）

| 项目 | 要求 |
|---|---|
| Python 依赖 | fastapi, sse-starlette, python-multipart, uvicorn, PyMuPDF, python-docx；MLX 还需 mlx, mlx-vlm |
| 翻译功能 (可选) | 任意 OpenAI 兼容 API (OpenAI / DeepSeek / 讯飞星火 / Ollama) |
| 浏览器 | Chrome / Safari / Firefox 现代浏览器 |

### 输出格式

每行一个 detection:
```
<|det|><type> [x1, y1, x2, y2]<|/det|><text>
```

- `<type>`: `title` / `text` / `header` / `table` / `[Non-Text]` 等
- `[x1, y1, x2, y2]`: 4-corner bounding box（归一化 0-1000 坐标）
- `<text>`: 识别出的文本

示例输出:
```markdown
<|det|>title [35, 38, 685, 88]<|/det|>Unlimited OCR 真实文档测试
<|det|>text [33, 114, 761, 150]<|/det|>百度于2026年6月22日发布 Unlimited-OCR 模型。
<|det|>text [33, 150, 576, 179]<|/det|>模型基于 Deepseek V2 架构,采用 MoE 路由机制。
```

### ❓ 常见问题

<details>
<summary><b>Web UI 启动报错 "ModuleNotFoundError"</b></summary>

```bash
# 确保激活了对应虚拟环境并重装依赖
source .venv-mlx/bin/activate
pip install -r web/requirements.txt
```
</details>

<details>
<summary><b>翻译按钮灰色 / 无法翻译</b></summary>

翻译需要配置 API Key。编辑 `web/.env`，修改后重启服务器生效。
</details>

<details>
<summary><b>模型加载很慢 / 内存不足</b></summary>

- 优先使用 MLX 量化版（峰值 ~5GB）
- 将模型放在内置 SSD（而非外置硬盘）
- 确保内存充足（≥24GB），避免频繁换页
</details>

<details>
<summary><b>OCR 识别结果有误，可以修改吗？</b></summary>

可以！Web UI 中所有识别文字都可以直接点击编辑，修改后自动保存，也会反映在导出的 Word / Markdown 中。
</details>

<details>
<summary><b>支持哪些文件格式？</b></summary>

- **上传**: PDF, PNG, JPG, JPEG, WebP, BMP
- **导出**: Markdown + 图片 (ZIP)、Word (.docx，仅原文 / 仅译文 / 双语对照)
</details>

### 已知限制

- **不官方支持**: 社区 patch，不是 Baidu 官方 Mac 路径
- **PyTorch/MPS 路径流式输出禁用**: MPS 有 "Placeholder storage" bug（PyTorch 2.12+），用 `eval_mode=True` 一次性返回
- **fp16 路径有 dtype mismatch**: 推荐 bf16
- **速度**: 比 NVIDIA GPU 慢 5-10x，但准确度高于 PaddleOCR 等传统 OCR

### 致谢

- [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) - 原始仓库 (MIT)
- [huggingface.co/baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) - 模型权重
- [sahilchachra/unlimited-ocr-mxfp8-mlx](https://huggingface.co/sahilchachra/unlimited-ocr-mxfp8-mlx) - MLX MXFP8 量化模型
- PyTorch MPS backend 团队

---

<a id="english"></a>

## English

Run [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) (6.7B VLM) **locally on Mac Apple Silicon (M1/M2/M3/M4)** — with a Web UI, real-time streaming, PDF native-text extraction, translation, and Word/Markdown export.

**Officially NVIDIA CUDA only.** This repo is a community-maintained Mac adaptation with two inference paths:

| Path | Description |
|---|---|
| 🚀 **MLX backend (default)** | Block float MX FP8 quantized model (~3.6 GB) + mlx-vlm streaming, ~205 tok/s, ~5 GB peak memory |
| 🧩 **PyTorch/MPS backend** | Official bf16 model (~6.7 GB) + 3 Mac patches, used by CLI |

### ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Local OCR** | 6.7B VLM runs entirely locally, no internet, privacy-safe |
| 🖥️ **Web UI** | Browser-based: Upload → Scan → Edit → Translate → Export |
| ⚡ **Real-time SSE** | Line-by-line streaming; pause / resume / stop / auto-follow |
| 📄 **PDF native text layer** | Digital PDFs use layout extraction (tables/formulas/figures restored) instead of the VLM — faster and more accurate; scanned pages still use the VLM |
| ✏️ **Inline Edit** | Click any text to edit, auto-saved into exports |
| 🌐 **Bilingual Translation** | 10+ languages, OpenAI-compatible APIs (OpenAI / DeepSeek / Ollama) |
| 📥 **Dual Export** | Markdown + images (ZIP) and Word (.docx, 3 modes) |
| 🤖 **MCP Service** | One-click integration with Claude Code / Codex / AI tools |

### Benchmarks (M4 Pro 48GB)

| Task | Time / Data |
|---|---|
| MLX MXFP8 decode | ~205 tok/s (FP16 baseline ~146 tok/s) |
| MLX peak memory / disk | ~5.0 GB / ~3.6 GB |
| PyTorch/MPS model load (6.7GB from SSD) | ~3s |
| Simple image (768×1024) | ~9s |
| Complex doc (1024×1024, 13 lines) | ~19s |

### 🚀 Install

**One-click install (recommended)**:

```bash
git clone https://github.com/Pzmpdm/unlimited-ocr-mac.git
cd unlimited-ocr-mac
bash install.sh            # MLX backend (default, recommended)
bash install.sh --pytorch  # or the PyTorch/MPS backend
bash install.sh --start    # auto-start the Web UI after install
```

Then start:

```bash
bash web/scripts/start_web.sh   # open http://localhost:8800
```

**Path A — MLX backend (manual, for reference)**

```bash
git clone https://github.com/Pzmpdm/unlimited-ocr-mac.git
cd unlimited-ocr-mac

# 1. Download the MLX quantized model (MXFP8, ~3.6 GB) and the original tokenizer
#    Model: https://huggingface.co/sahilchachra/unlimited-ocr-mxfp8-mlx
#    Place under ../Unlimited-OCR-MLX and ../Unlimited-OCR-model (original bf16 weights, tokenizer.json only)
#    Override with OCR_MLX_MODEL_DIR if needed

# 2. Create venv and install deps
python3.13 -m venv .venv-mlx
source .venv-mlx/bin/activate
pip install mlx mlx-vlm transformers \
    fastapi sse-starlette python-multipart uvicorn PyMuPDF python-docx

# 3. Start the Web UI
bash web/scripts/start_web.sh        # default port 8800
# Open http://localhost:8800
```

**Path B — PyTorch/MPS backend (CLI / fallback)**

```bash
git clone https://github.com/Pzmpdm/unlimited-ocr-mac.git
cd unlimited-ocr-mac
bash scripts/install.sh          # Install OCR model (~5-10 min, 6.7GB download, auto-patched)
bash web/scripts/install_web.sh  # Install Web UI deps (~1 min)
```

### One-click run

**Web UI (recommended)**:

```bash
bash web/scripts/start_web.sh
# Open http://localhost:8800
# MLX default; switch to PyTorch: OCR_BACKEND=pytorch bash web/scripts/start_web.sh
```

**CLI (PyTorch/MPS backend)**:

```bash
bash scripts/run.sh /path/to/your.png
bash scripts/run.sh --image_dir /path/to/images/ --output_dir ./out
```

### MCP service (for Claude Code / Codex / other AI tools)

```bash
bash scripts/mcp.sh    # start stdio MCP server
```

Register with Claude Code (`~/.claude.json` or project `.mcp.json`):
```json
{
  "mcpServers": {
    "unlimited-ocr": {
      "command": "python3",
      "args": ["/path/to/unlimited-ocr-mac/mcp_server/server.py"]
    }
  }
}
```

Register with OpenAI Codex CLI (`~/.codex/config.toml`):
```toml
[mcp_servers.unlimited-ocr]
command = "python3"
args = ["/abs/path/to/unlimited-ocr-mac/mcp_server/server.py"]
```

Tools exposed: `ocr_image(path)`, `ocr_directory(path)`, `model_status()`.

### 🏗️ Project Structure

```
unlimited-ocr-mac/
├── patches/              # Mac adaptation patches
├── scripts/              # CLI install/run scripts (install.sh / run.sh / mcp.sh)
├── mcp_server/           # MCP Server (server.py)
├── web/                  # 🌐 Web UI (FastAPI + SSE, MLX backend default)
│   ├── server.py         #   main server
│   ├── ocr_engine_mlx.py #   MLX engine (mlx-vlm streaming)
│   ├── ocr_engine.py     #   PyTorch/MPS engine (lazy load)
│   ├── ocr_parser.py     #   parse <|det|> output → HTML/Markdown
│   ├── pdf_converter.py  #   PDF render + native text/table/formula/figure extraction
│   ├── translator.py     #   translation (OpenAI-compatible)
│   ├── docx_exporter.py  #   Word export
│   ├── public/           #   frontend (index.html / style.css / app.js)
│   └── scripts/          #   install_web.sh / start_web.sh
├── install.sh            # one-click installer (MLX default / --pytorch)
├── run_mac.py            # Mac adaptation core (apply patches + inference)
└── README.md
```

### Two Mac adaptation approaches

**1. MLX quantization (default, recommended)**

Model `sahilchachra/unlimited-ocr-mxfp8-mlx` (Block float MX FP8, 9.19 effective bits/weight, ~3.6 GB), run with mlx-vlm `stream_generate`; decoded with Baidu's original `tokenizer.json`; built-in sliding n-gram dedup (ngram=35, window=128); live tok/s and peak-memory stats in the UI.

**2. PyTorch/MPS — three patches (CLI / fallback)**

1. `.cuda()` hardcoded → `.to('mps')` (13+ places in `modeling_unlimitedocr.py`; rewritten automatically by `run_mac.py::apply_patches()`)
2. Disable `torch.autocast("cuda", ...)` ⭐ **most critical** — MPS autocast bf16 causes MLA+MoE precision drift (logits collapse into `168168168...` loops); removing it fixes inference
3. `SlidingWindowNoRepeatNgramProcessor` must inherit `LogitsProcessor` so transformers 4.57 actually calls it

Patches live in `patches/`; `install.sh` overlays them on the downloaded HF model.

### 🔧 System Requirements

| Item | Requirement |
|---|---|
| OS | macOS 14+ (Sonoma) on Apple Silicon (M1/M2/M3/M4) |
| Python | 3.12 or 3.13 (3.13 recommended) |
| Unified Memory | **≥ 24GB** (32GB+ recommended; 48GB tested) |
| Disk | ~4GB (MLX) / ~8GB (PyTorch) |

### Output format

One line per detection:
```
<|det|><type> [x1, y1, x2, y2]<|/det|><text>
```

- `<type>`: `title` / `text` / `header` / `table` / `[Non-Text]` etc.
- `[x1, y1, x2, y2]`: normalized 0-1000 bounding box (pixels)

Example:
```markdown
<|det|>title [35, 38, 685, 88]<|/det|>Unlimited OCR Real Document Test
<|det|>text [33, 114, 761, 150]<|/det|>Baidu released Unlimited-OCR model on 2026-06-22.
```

### ❓ FAQ

- **Web UI "ModuleNotFoundError"**: activate the matching venv and reinstall deps: `source .venv-mlx/bin/activate && pip install -r web/requirements.txt`
- **Translate button grayed out**: configure `web/.env` (`TRANSLATE_API_BASE` / `TRANSLATE_API_KEY` / `TRANSLATE_MODEL`), restart the server
- **Model loading slow / low memory**: use the MLX quantized model (~5 GB peak); put the model on internal SSD; ≥24 GB RAM
- **Can I edit OCR results?**: Yes — click any text in the Web UI; edits are saved and reflected in exports
- **Supported formats**: upload PDF/PNG/JPG/JPEG/WebP/BMP; export Markdown+images (ZIP) / Word (.docx)

### Known limitations

- **Not officially supported** — community patches, not Baidu's official Mac path
- **PyTorch/MPS streaming disabled** — MPS "Placeholder storage" bug; use `eval_mode=True` (returns the full string at once)
- **fp16 path has dtype mismatch** — use bf16
- **Speed** ~20s/image (PyTorch/MPS), 5-10× slower than NVIDIA GPU but more accurate than traditional OCR (PaddleOCR, etc.)

### Credits

- [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) — original repo (MIT)
- [huggingface.co/baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) — model weights
- [sahilchachra/unlimited-ocr-mxfp8-mlx](https://huggingface.co/sahilchachra/unlimited-ocr-mxfp8-mlx) — MLX MXFP8 quantized model
- PyTorch MPS backend team