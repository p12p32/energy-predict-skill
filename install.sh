#!/bin/bash
# install.sh — 一键安装电力预测 Skill
# 自动适配 zsh / bash / fish, 处理依赖缺失, 支持无 root 安装
set -e

echo "=== 电力预测 Skill 安装 ==="
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 自动探测 Python / Pip
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "python3")
PIP=$(command -v pip3 2>/dev/null || command -v pip 2>/dev/null || echo "pip3")

# ── Shell 探测 ──
detect_shell_rc() {
    local shell_name
    shell_name="$(basename "$SHELL" 2>/dev/null)"
    case "$shell_name" in
        zsh)  echo "$HOME/.zshrc" ;;
        bash) echo "$HOME/.bashrc" ;;
        fish) echo "$HOME/.config/fish/config.fish" ;;
        *)    echo "$HOME/.bashrc" ;;
    esac
}
RC_FILE=$(detect_shell_rc)
mkdir -p "$(dirname "$RC_FILE")" 2>/dev/null

# ── 0. 环境预检 ──
echo "[0/3] 检测编译环境..."
NEEDS_LGB=true
if ! command -v gcc &>/dev/null && ! command -v clang &>/dev/null; then
    echo "  ⚠ 未检测到 C 编译器 (gcc/clang)"
    echo "    LightGBM 需要编译, 跳过安装"
    echo "    可先安装: brew install gcc (macOS) 或 apt install build-essential (Linux)"
    echo "    模型预测功能不受影响(仅训练需要 LightGBM)"
    NEEDS_LGB=false
fi

# ── 1. 安装依赖 ──
echo ""
echo "[1/3] 安装 Python 依赖..."

# 基础包 (无需编译, 快速)
BASE_PKGS="pandas numpy pymysql pyyaml scikit-learn requests pyarrow pytest"
echo "  安装基础包..."
$PIP install --user $BASE_PKGS 2>/dev/null || $PIP install $BASE_PKGS 2>/dev/null || {
    echo "  ⚠ $PIP 失败, 尝试 pip..."
    pip install --user $BASE_PKGS 2>/dev/null || pip install $BASE_PKGS
}

# LightGBM (需编译, 可选)
if [ "$NEEDS_LGB" = true ]; then
    echo "  安装 LightGBM (编译中, 约 1-3 分钟)..."
    $PIP install --user lightgbm 2>/dev/null || $PIP install lightgbm 2>/dev/null || {
        echo "  ⚠ LightGBM 编译失败"
        echo "    可手动安装: brew install libomp && $PIP install lightgbm (macOS)"
        echo "    或: $PIP install lightgbm --no-cache-dir (重试)"
        echo "    预测功能仍可用, 训练需要 LightGBM"
    }
fi

# Prophet (可选)
echo "  安装 Prophet..."
$PIP install --user prophet 2>/dev/null || $PIP install prophet 2>/dev/null || echo "  Prophet 跳过(可选, 不影响核心功能)"

echo "  依赖安装完成"

# ── 2. 环境变量 ──
echo ""
echo "[2/3] 设置环境变量..."

if ! grep -q "ENERGY_HOME" "$RC_FILE" 2>/dev/null; then
  echo "export ENERGY_HOME=\"$SCRIPT_DIR\"" >> "$RC_FILE"
fi
if ! grep -q "energy-predict" "$RC_FILE" 2>/dev/null; then
  echo "alias energy-predict=\"$SCRIPT_DIR/energy-predict\"" >> "$RC_FILE"
fi
echo "  已写入: $RC_FILE"

# ── 3. 验证 ──
echo ""
echo "[3/3] 验证安装..."
chmod +x "$SCRIPT_DIR/energy-predict"
export ENERGY_HOME="$SCRIPT_DIR"

$PYTHON -c "from src.core.config import load_config; load_config(); print('  ✅ 配置加载')" 2>/dev/null || echo "  ⚠ 配置加载失败 (检查 config.yaml)"
$PYTHON -c "import pandas, numpy; print(f'  ✅ pandas {pandas.__version__}, numpy {numpy.__version__}')" 2>/dev/null || echo "  ⚠ pandas/numpy 未安装"
$PYTHON -c "import lightgbm; print(f'  ✅ LightGBM {lightgbm.__version__}')" 2>/dev/null || echo "  ⚠ LightGBM 未安装 (轻量模式, 无训练功能)"
$PYTHON -c "import requests; print('  ✅ requests OK')" 2>/dev/null || echo "  ⚠ requests 未安装"

echo ""
echo "============================================"
echo "  安装完成！"
echo ""
echo "  执行生效:"
echo "    source $RC_FILE"
echo ""
echo "  快速开始:"
echo "    energy-predict import data/your_data.csv"
echo "    energy-predict predict 广东 load 24"
echo "    energy-predict benchmark"
echo "============================================"
