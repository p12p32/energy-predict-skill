#!/bin/bash
# install.sh — 一键安装电力预测 Skill
set -e

echo "=== 电力预测 Skill 安装 ==="
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/3] 安装 Python 依赖..."
pip3 install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "[2/3] 设置环境变量..."
if ! grep -q "ENERGY_HOME" ~/.zshrc 2>/dev/null; then
  echo "export ENERGY_HOME=\"$SCRIPT_DIR\"" >> ~/.zshrc
fi
if ! grep -q "energy-predict" ~/.zshrc 2>/dev/null; then
  echo "alias energy-predict=\"$SCRIPT_DIR/energy-predict\"" >> ~/.zshrc
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
echo "    source ~/.zshrc"
echo ""
echo "  快速开始:"
echo "    energy-predict import data/your_data.csv"
echo "    energy-predict predict 广东 load 24"
echo "    energy-predict benchmark"
echo "============================================"
