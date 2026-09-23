"""Adapters for public pre-trained music models (protocol, "M-WILD").

Probing and editing a public checkpoint needs two model-specific things: its token
scheme (how a score becomes token ids, and where in the stream the model chooses a
pitch) and its layer access (which module to hook, and how to read the residual
stream). Everything else in the protocol is shared. Each public model therefore
contributes one adapter and no changes to the experiment scripts.
"""

from src.publicmodels.base import PublicModelAdapter
from src.publicmodels.registry import ADAPTERS, get_adapter

__all__ = ["PublicModelAdapter", "ADAPTERS", "get_adapter"]
