from __future__ import annotations

from collections.abc import Callable


def _always_false() -> bool:
    return False


def _replace_cached_flag(flag: Callable[[], bool], replacement: Callable[[], bool]) -> Callable[[], bool]:
    cache_clear = getattr(flag, 'cache_clear', None)
    if cache_clear is not None:
        cache_clear()
    return replacement


def disable_optional_vision_backends() -> None:
    """Keep optional vision packages from blocking text-only Transformer models."""
    import transformers
    import transformers.utils as transformers_utils
    from transformers.utils import import_utils

    import_utils.is_torchvision_available = _replace_cached_flag(
        import_utils.is_torchvision_available,
        _always_false,
    )
    import_utils.is_torchvision_v2_available = _replace_cached_flag(
        import_utils.is_torchvision_v2_available,
        _always_false,
    )
    transformers_utils.is_torchvision_available = import_utils.is_torchvision_available
    transformers_utils.is_torchvision_v2_available = import_utils.is_torchvision_v2_available
    transformers.is_torchvision_available = import_utils.is_torchvision_available


disable_optional_vision_backends()

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)

__all__ = [
    'AutoModelForSeq2SeqLM',
    'AutoTokenizer',
    'DataCollatorForSeq2Seq',
    'get_linear_schedule_with_warmup',
]
