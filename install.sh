#!/bin/bash
# install.sh — 一键安装电力预测 Skill
# 自动适配 zsh / bash / fish
set -e

echo "=== 电力预测 Skill 安装 ==="
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/3] 安装 Python 依赖..."
pip3 install -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null || pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "[2/3] 设置环境变量..."

# 探测 shell → 选对应的 rc 文件
detect_shell_rc() {
    local shell_name
    shell_name="$(basename "$SHELL" 2>/dev/null)"

    case "$shell_name" in
        zsh)  echo "$HOME/.zshrc" ;;
        bash) echo "$HOME/.bashrc" ;;
        fish) echo "$HOME/.config/fish/config.fish" ;;
        *)    echo "$HOME/.bashrc" ;;  # fallback
    esac
}

RC_FILE=$(detect_shell_rc)
RC_DIR=$(dirname "$RC_FILE")
mkdir -p "$RC_DIR" 2>/dev/null

if ! grep -q "ENERGY_HOME" "$RC_FILE" 2>/dev/null; then
  echo "export ENERGY_HOME=\"$SCRIPT_DIR\"" >> "$RC_FILE"
fi

if ! grep -q "energy-predict" "$RC_FILE" 2>/dev/null; then
  echo "alias energy-predict=\"$SCRIPT_DIR/energy-predict\"" >> "$RC_FILE"
fi

echo "  已写入: $RC_FILE"

echo ""
echo "[3/3] 验证安装..."
chmod +x "$SCRIPT_DIR/energy-predict"
export ENERGY_HOME="$SCRIPT_DIR"
python3 -c "from src.core.config import load_config; load_config(); print('配置加载 OK')"
python3 -c "import lightgbm; print('LightGBM', lightgbm.__version__, 'OK')" 2>/dev/null || echo "LightGBM 将在首次训练时编译"

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
