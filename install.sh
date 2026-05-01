#!/bin/bash
# install.sh — 一键安装电力预测 Skill
set -e

echo "=== 电力预测 Skill 安装 ==="
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 检测 Python 命令 ──
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "错误: 未找到 Python，请先安装 Python >= 3.10"
    exit 1
fi
echo "检测到 Python: $PYTHON ($($PYTHON --version 2>&1))"

# ── 检测 Shell RC 文件 ──
detect_rc_file() {
    local shell_name
    shell_name="$(basename "$SHELL" 2>/dev/null)"
    case "$shell_name" in
        zsh)  echo "$HOME/.zshrc" ;;
        bash) echo "$HOME/.bashrc" ;;
        *)
            # 回退: 检查实际存在的文件
            for f in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.bash_profile" "$HOME/.profile"; do
                if [ -f "$f" ] || [ "$f" = "$HOME/.bashrc" ]; then
                    echo "$f"
                    return
                fi
            done
            echo "$HOME/.bashrc"
            ;;
    esac
}

RC_FILE=$(detect_rc_file)
echo "目标 Shell 配置: $RC_FILE"

echo ""
echo "[1/3] 安装 Python 依赖..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "[2/3] 设置环境变量..."
if ! grep -q "ENERGY_HOME" "$RC_FILE" 2>/dev/null; then
    echo "export ENERGY_HOME=\"$SCRIPT_DIR\"" >> "$RC_FILE"
fi
if ! grep -q "energy-predict" "$RC_FILE" 2>/dev/null; then
    echo "alias energy-predict=\"$SCRIPT_DIR/energy-predict\"" >> "$RC_FILE"
fi

echo ""
echo "[3/3] 验证安装..."
chmod +x "$SCRIPT_DIR/energy-predict"
$PYTHON -c "from src.core.config import load_config; load_config(); print('配置加载 OK')"
$PYTHON -c "import lightgbm; print('LightGBM', lightgbm.__version__, 'OK')"

echo ""
echo "============================================"
echo "  安装完成！"
echo ""
echo "  重新打开终端，或执行:"
echo "    source $RC_FILE"
echo ""
echo "  快速开始:"
echo "    energy-predict import data/your_data.csv"
echo "    energy-predict predict 广东 load 24"
echo "    energy-predict benchmark"
echo "============================================"
