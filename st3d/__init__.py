"""
st3d: 3D spatial transcriptomics reconstruction from serial 2D sections.

Models tissue continuity along the z-axis with neural ODEs (Euler integration)
and local spatial attention, using only standard PyTorch with an AnnData-native
pipeline.
"""

__version__ = "0.1.0"

from st3d.config import ST3DConfig
from st3d.model import ST3DModel
from st3d.data import preprocess_sections
from st3d.training import Trainer
from st3d.inference import reconstruct_volume

__all__ = [
    "ST3DConfig",
    "ST3DModel",
    "preprocess_sections",
    "Trainer",
    "reconstruct_volume",
    "__version__",
]
