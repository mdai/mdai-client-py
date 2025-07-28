"""MD.ai Python client library."""

from importlib import metadata

__version__ = metadata.version("mdai")

from . import preprocess
from .client import Client
from .utils import common_utils
from .utils import transforms
from .utils import dicom_utils
from .inference import delete_env, run_inference, infer

try:
    CAN_VISUALIZE = True
    from . import visualize
except ImportError:
    # matplotlib backend missing or cannot be loaded
    CAN_VISUALIZE = False
