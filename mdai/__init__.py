"""MD.ai Python client library."""

__version__ = "0.2.1"

from . import preprocess
from .client import Client
from .utils import common_utils

try:
    CAN_VISUALIZE = True
    from . import visualize
except ImportError:
    # matplotlib backend missing or cannot be loaded
    CAN_VISUALIZE = False
