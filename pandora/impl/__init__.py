try:
    from pandora.impl.keras.model import KerasModel
except ImportError:
    print("Couldn't import Keras. Keras implementation not available")
    KerasModel = None
try:
    from pandora.impl.pytorch.model import PyTorchModel
except ImportError:
    print("Couldn't import PyTorch. PyTorch implementation not available")
    PyTorchModel = None
