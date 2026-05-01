#!/bin/bash
# install.sh — 一键安装电力预测 Skill
# 自动检测 zsh/bash 并写入对应配置文件
set -e

echo "=== 电力预测 Skill 安装 ==="
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 检测 shell ──
detect_shell_rc() {
    local shell_name
    shell_name=$(basename "$SHELL" 2>/dev/null)
    case "$shell_name" in
        zsh)  echo "$HOME/.zshrc" ;;
        bash) echo "$HOME/.bashrc" ;;
        *)    echo "$HOME/.profile" ;;
    esac
}

SHELL_RC=$(detect_shell_rc)
echo "检测到 shell: $(basename "$SHELL") → 配置文件: $SHELL_RC"

echo ""
echo "[1/3] 安装 Python 依赖..."
pip3 install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "[2/3] 设置环境变量..."
# 确保配置文件存在
touch "$SHELL_RC"

if ! grep -q "ENERGY_HOME" "$SHELL_RC" 2>/dev/null; then
    echo "export ENERGY_HOME=\"$SCRIPT_DIR\"" >> "$SHELL_RC"
    echo "  写入: export ENERGY_HOME=\"$SCRIPT_DIR\""
else
    echo "  ENERGY_HOME 已存在，跳过"
fi

if ! grep -q "energy-predict" "$SHELL_RC" 2>/dev/null; then
    echo "alias energy-predict=\"$SCRIPT_DIR/energy-predict\"" >> "$SHELL_RC"
    echo "  写入: alias energy-predict=\"$SCRIPT_DIR/energy-predict\""
else
    echo "  energy-predict 别名已存在，跳过"
fi

echo ""
echo "[3/3] 验证安装..."
chmod +x "$SCRIPT_DIR/energy-predict"
python3 -c "from src.core.config import load_config; load_config(); print('配置加载 OK')"
python3 -c "import lightgbm; print('LightGBM', lightgbm.__version__, 'OK')"

echo ""
echo "============================================"
echo "  安装完成！"
echo ""
echo "  重新打开终端，或执行:"
echo "    source $SHELL_RC"
echo ""
echo "  快速开始:"
echo "    energy-predict import data/your_data.csv"
echo "    energy-predict predict 广东 load 24"
echo "    energy-predict benchmark"
echo "============================================"
