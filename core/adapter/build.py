from .base_adapter import BaseAdapter
from .rotta import RoTTA
from .dynamic_rotta import DynamicMemorySizeRoTTA


def build_adapter(cfg):
    if cfg.ADAPTER.NAME == "rotta":
        return RoTTA
    elif cfg.ADAPTER.NAME == "dynamic_rotta":
        return DynamicMemorySizeRoTTA
    else:
        raise ValueError(f"Unknown adapter: {cfg.ADAPTER.NAME}")