from .transformers import (
    BaseTransformer,
    Log10Transformer,
    RowWiseMinMaxScaler,
    NormalScoreTransformer,
    TransformerPipeline,
    AutobotsAssemble
)
import importlib.util
from .base import Emulator
from .dsi import DSI
from .dsivc import DSIVC


__all__ = [
    'Emulator',
    'DSI',
    'DSIVC',
    'LPFA',
    'GPR',
    'DSIAE',
    'BaseTransformer',
    'Log10Transformer',
    'RowWiseMinMaxScaler',
    'StandardScalerTransformer',
    'NormalScoreTransformer',
    'TransformerPipeline',
    'AutobotsAssemble'
]

# Optional-dependency availability
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

try:
    import tensorflow
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


# Conditional imports
if HAS_SKLEARN:
    from .lpfa import LPFA
    from .gpr import GPR
    from .transformers import StandardScalerTransformer
else:
    # Create placeholder classes that raise informative errors
    class LPFA:
        def __init__(self, *args, **kwargs):
            raise ImportError("LPFA emulator requires scikit-learn. Install with: pip install scikit-learn")

    class GPR:
        def __init__(self, *args, **kwargs):
            raise ImportError("GPR emulator requires scikit-learn. Install with: pip install scikit-learn")

    class StandardScalerTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("StandardScalerTransformer requires scikit-learn. Install with: pip install scikit-learn")

if HAS_TENSORFLOW and HAS_SKLEARN:
    from .dsiae import DSIAE
else:
    class DSIAE:
        def __init__(self, *args, **kwargs):
            raise ImportError("DSIAE emulator requires TensorFlow. Install with: pip install tensorflow")
