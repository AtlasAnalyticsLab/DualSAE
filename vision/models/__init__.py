"""Sparse autoencoder models used in the paper.

Baselines (TopK, BatchTopK, JumpReLU, vanilla ReLU+L1, Matryoshka, MSAE, Gated)
and DualSAE, whose Sparse Branch can use any of those sparsification rules.
"""

from .topk_sae import TopKSAE, TopK, TiedTranspose, ACTIVATIONS_CLASSES
from .batch_topk_sae import BatchTopKSAE
from .vanilla_sae import VanillaSAE
from .jumprelu_sae import JumpReLUSAE
from .gated_sae import GatedSAE
from .dual_sae import DualSAE
from .matryoshka_sae import MatryoshkaSAE
from .msae_sae import MSAE
from .base import BaseAutoencoder

__all__ = [
    'TopKSAE', 'TopK', 'TiedTranspose', 'ACTIVATIONS_CLASSES', 'BatchTopKSAE',
    'VanillaSAE', 'JumpReLUSAE', 'GatedSAE', 'DualSAE', 'MatryoshkaSAE', 'MSAE',
    'BaseAutoencoder',
]
