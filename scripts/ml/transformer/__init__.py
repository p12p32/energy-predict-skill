"""transformer — Transformer 残差修正 (可选 PyTorch 依赖).

物理模型预测主体信号, Transformer 学习残差模式并做滚动实时修正.
"""
from scripts.ml.transformer.corrector import ResidualTransformer, PositionalEncoding
from scripts.ml.transformer.trainer import TransformerTrainer

__all__ = ["ResidualTransformer", "PositionalEncoding", "TransformerTrainer"]
