# 电力预测 Skill

3 步预测电力出力/负荷/电价，模型自动进化。

## 安装

```bash
git clone <repo-url> && cd energy-predict-skill
bash assets/install.sh
source ~/.zshrc
```

## 3 步上手

```bash
# 1. 导入 CSV
energy-predict import data/guangdong.csv

# 2. 预测
energy-predict predict 广东 load 24

# 3. 看精度
energy-predict benchmark
```

## CSV 格式

```csv
dt,province,type,value,price
2025-01-01 00:00:00,广东,load,48000,0.35
2025-01-01 00:15:00,广东,load,47800,0.35
```

必需列: `dt`, `province`, `type`, `value`
可选列: `price`

## 命令

| 命令 | 说明 |
|------|------|
| `energy-predict import <csv>` | 导入数据 |
| `energy-predict predict <省> <类型> <小时>` | 预测 |
| `energy-predict status` | 模型状态 |
| `energy-predict explain <省> <类型>` | 特征重要性 |
| `energy-predict export <省> <类型> <格式>` | 导出 |
| `energy-predict benchmark` | 精度测试 |
| `energy-predict daemon [分钟]` | 自动运行 |

## 作为 OpenCode Skill

将 `SKILL.md` 复制到你的 OpenCode skills 目录即可自动激活。
