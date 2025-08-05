"""
models package for the Blood Flow Estimation project.

Provides access to core model architectures used for
predicting blood flow from speckle video sequences.

Modules:
- bloodflow_cnn: 3D CNN model for continuous blood flow rate estimation.
"""

from .bloodflow_cnn import BloodFlowCNN

__all__ = [
    "BloodFlowCNN",
]