#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#   Unlimited OCR · Mac 一键安装脚本  (Apple Silicon)
#
#   默认安装 MLX 后端（推荐）：MXFP8 量化模型 ~3.6 GB，峰值内存 ~5 GB
#   --pytorch 安装 PyTorch/MPS 后端：官方 bf16 模型 ~6.7 GB + 3 个补丁
#
#   用法 / Usage:
#     bash install.sh                  # 一键安装 MLX 后端（默认，推荐）
#     bash install.sh --pytorch        # 安装 PyTorch/MPS 后端
#     bash install.sh --mlx-dir PATH   # 自定义 MLX 模型目录
#     bash install.sh --start          # 安装完成后自动启动 Web UI
#     bash install.sh --help           # 显示帮助
#
#   环境变量 / Env:
#     OCR_MLX_MODEL_DIR   MLX 模型目录（默认 <仓库上级>/Unlimited-OCR-MLX）
#     HF_HUB_CACHE        HF 缓存目录（可指向外置 SSD）
#     HF_ENDPOINT         HF 镜像（国内可设 https://hf-mirror.com）
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$REPO_DIR")"

BACKEND="mlx"
START_WEB=0
MLX_MODEL_DIR="${OCR_MLX_MODEL_DIR:-$PARENT_DIR/Unlimited-OCR-MLX}"
ORIG_MODEL_DIR="$PARENT_DIR/Unlimited-OCR-model"

# ── 日志 ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "$GREEN[INFO]$NC $*"; }
warn() { echo -e "$YELLOW[WARN]$NC $*"; }
err()  { echo -e "$RED[ERR]$NC $*" >&2; }
step() { echo -e "$CYAN[STEP]$NC $*"; }
die()  { err "$*"; exit 1; }
trap 'err "安装失败，请查看上方错误信息。"; exit 1' ERR

usage() {
    cat <<'EOF'
用法 / Usage:
  bash install.sh                  # 一键安装 MLX 后端（默认，推荐）
  bash install.sh --pytorch        # 安装 PyTorch/MPS 后端
  bash install.sh --mlx-dir PATH   # 自定义 MLX 模型目录
  bash install.sh --start          # 安装完成后自动启动 Web UI
  bash install.sh --help           # 显示帮助

环境变量 / Env:
  OCR_MLX_MODEL_DIR   MLX 模型目录（默认 <仓库上级>/Unlimited-OCR-MLX）
  HF_HUB_CACHE        HF 缓存目录（可指向外置 SSD）
  HF_ENDPOINT         HF 镜像（国内可设 https://hf-mirror.com）
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pytorch)  BACKEND="pytorch"; shift ;;
        --mlx-dir)  [[ $# -ge 2 ]] || die "--mlx-dir 需要一个路径参数"; MLX_MODEL_DIR="$2"; shift 2 ;;
        --start)    START_WEB=1; shift ;;
        --help|-h)  usage; exit 0 ;;
        *) die "未知参数: $1（--help 查看用法）" ;;
    esac
done

# ── 1. 环境检查 ───────────────────────────────────────────────────
step "检查环境 ..."
if [[ "$(uname -s)" != "Darwin" ]]; then
    warn "非 macOS：MLX / MPS 不可用，将退回 CPU（极慢）。"
fi
if [[ "$(uname -m)" != "arm64" ]]; then
    warn "非 Apple Silicon (arm64)：MLX / MPS 不可用，将退回 CPU（极慢）。"
fi

PYTHON_BIN=""
for cand in python3.13 python3.12 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYTHON_BIN="$cand"
        break
    fi
done
[[ -z "$PYTHON_BIN" ]] && die "未找到 Python 3，请先安装：brew install python@3.13"
PY_MAJOR="$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.major)')"
PY_MINOR="$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.minor)')"
if [[ "$PY_MAJOR" -lt 3 || "$PY_MINOR" -lt 12 ]]; then
    die "需要 Python 3.12+，当前 $PYTHON_BIN $PY_MAJOR.$PY_MINOR（brew install python@3.13）"
fi
info "Python: $PYTHON_BIN ($PY_MAJOR.$PY_MINOR)"

# ── 2. 虚拟环境 ───────────────────────────────────────────────────
if [[ "$BACKEND" == "mlx" ]]; then
    VENV_DIR="$REPO_DIR/.venv-mlx"
else
    VENV_DIR="$REPO_DIR/.venv-ocr"
fi
step "后端: $BACKEND  |  虚拟环境: $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
    info "创建虚拟环境 $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    info "复用已有虚拟环境 $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── 3. 磁盘空间提示 ───────────────────────────────────────────────
NEED_GB=5
[[ "$BACKEND" == "pytorch" ]] && NEED_GB=10
AVAIL_KB="$(df -k "$REPO_DIR" | awk 'NR==2 {print $4}')"
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if [[ "$AVAIL_GB" -lt "$NEED_GB" ]]; then
    warn "磁盘可用空间仅 $AVAIL_GB GB（建议 ≥$NEED_GB GB），模型下载可能失败"
fi

# ── 4. 安装依赖 ───────────────────────────────────────────────────
if [[ "$BACKEND" == "mlx" ]]; then
    step "安装 MLX 依赖（mlx / mlx-vlm / transformers + Web 依赖）..."
    pip install --quiet --upgrade pip
    pip install mlx "mlx-vlm>=0.6" transformers huggingface_hub \
        Pillow PyMuPDF python-docx \
        fastapi sse-starlette python-multipart uvicorn
else
    step "PyTorch/MPS 后端：调用 scripts/install.sh（下载 bf16 模型 ~6.7 GB + 打补丁）..."
    bash "$REPO_DIR/scripts/install.sh"
    step "安装 Web 依赖 ..."
    pip install --quiet fastapi sse-starlette python-multipart uvicorn \
        "python-docx>=1.1.0" PyMuPDF
fi

# ── 5. 下载模型（MLX 后端）───────────────────────────────────────
if [[ "$BACKEND" == "mlx" ]]; then
    step "检查 MLX 模型 $MLX_MODEL_DIR ..."
    if [[ -f "$MLX_MODEL_DIR/model.safetensors" ]]; then
        info "模型已存在，跳过下载：$MLX_MODEL_DIR"
    else
        info "下载 MLX 量化模型（MXFP8, ~3.6 GB）到 $MLX_MODEL_DIR ..."
        mkdir -p "$MLX_MODEL_DIR"
        MLX_MODEL_DIR="$MLX_MODEL_DIR" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id="sahilchachra/unlimited-ocr-mxfp8-mlx", local_dir=os.environ["MLX_MODEL_DIR"])
PY
        info "MLX 模型下载完成"
    fi

    step "检查原始 tokenizer $ORIG_MODEL_DIR/tokenizer.json ..."
    if [[ -f "$ORIG_MODEL_DIR/tokenizer.json" ]]; then
        info "tokenizer 已存在，跳过下载"
    else
        info "下载原始 tokenizer（baidu/Unlimited-OCR）..."
        mkdir -p "$ORIG_MODEL_DIR"
        ORIG_MODEL_DIR="$ORIG_MODEL_DIR" python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="baidu/Unlimited-OCR", filename="tokenizer.json", local_dir=os.environ["ORIG_MODEL_DIR"])
PY
        info "tokenizer 下载完成"
    fi
fi

# ── 6. 生成 web/.env ──────────────────────────────────────────────
step "配置 web/.env ..."
if [[ ! -f "$REPO_DIR/web/.env" ]]; then
    cp "$REPO_DIR/web/.env.example" "$REPO_DIR/web/.env"
    info "已从模板生成 web/.env（如需翻译功能请填入 API Key）"
else
    info "web/.env 已存在，保留现有配置"
fi

# ── 7. 验证安装 ───────────────────────────────────────────────────
step "验证安装 ..."
if [[ "$BACKEND" == "mlx" ]]; then
    python3 -c 'import mlx_vlm, transformers, fastapi, uvicorn, pymupdf, docx; print("依赖 OK")'
    [[ -f "$MLX_MODEL_DIR/model.safetensors" ]] || die "MLX 模型文件缺失: $MLX_MODEL_DIR"
    [[ -f "$ORIG_MODEL_DIR/tokenizer.json" ]] || die "tokenizer 缺失: $ORIG_MODEL_DIR/tokenizer.json"
    info "MLX 模型: $MLX_MODEL_DIR"
else
    python3 -c 'import torch, transformers, fastapi, uvicorn, pymupdf, docx; print("依赖 OK")'
    [[ -f "$REPO_DIR/model_dir/config.json" ]] || die "模型目录缺失: $REPO_DIR/model_dir（请检查 scripts/install.sh）"
    info "模型目录: $REPO_DIR/model_dir"
fi

# ── 8. 完成 ───────────────────────────────────────────────────────
if [[ -z "${HF_ENDPOINT:-}" ]]; then
    warn "提示：国内网络下载 HF 模型较慢，可先执行 export HF_ENDPOINT=https://hf-mirror.com 再重装。"
fi
echo ""
echo "════════════════════════════════════════════════════════════"
echo -e "$GREEN  ✓ 安装完成！$NC"
echo "════════════════════════════════════════════════════════════"
echo ""
if [[ "$BACKEND" == "mlx" ]]; then
    echo "  启动 Web UI（MLX 后端）:"
    echo "    bash web/scripts/start_web.sh"
else
    echo "  启动 Web UI（PyTorch 后端）:"
    echo "    OCR_BACKEND=pytorch bash web/scripts/start_web.sh"
    echo "  或命令行 OCR:"
    echo "    bash scripts/run.sh /path/to/image.png"
fi
echo "  浏览器打开: http://localhost:8800"
echo ""
if [[ "$START_WEB" == "1" ]]; then
    step "启动 Web UI ..."
    if [[ "$BACKEND" == "mlx" ]]; then
        exec bash "$REPO_DIR/web/scripts/start_web.sh"
    else
        cd "$REPO_DIR/web" && exec env OCR_BACKEND=pytorch "$VENV_DIR/bin/python" server.py
    fi
fi
exit 0
