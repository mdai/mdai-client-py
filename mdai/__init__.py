"""MD.ai Python client library."""

try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata
__version__ = metadata.version("mdai")

from . import preprocess
from .client import Client
from .utils import common_utils
from .utils import transforms
from .inference import delete_env, run_inference, infer

try:
    CAN_VISUALIZE = True
    from . import visualize
except ImportError:
    # matplotlib backend missing or cannot be loaded
    CAN_VISUALIZE = False
