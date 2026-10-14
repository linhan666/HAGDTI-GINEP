"""Compatibility helpers for EvolutionaryScale ESMC tokenization."""


_ESMC_SPECIAL_TOKENS = {
    "pad_token":  "<pad>",
    "cls_token":  "<cls>",
    "eos_token":  "<eos>",
    "mask_token": "<mask>",
    "bos_token":  "<cls>",
    "unk_token":  "<unk>",
}


def _get_esmc_tokenizer_cls():
    try:
        from esm.tokenization import EsmSequenceTokenizer
    except ImportError:
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    return EsmSequenceTokenizer


def _patch_esmc_tokenizer_cls(tokenizer_cls):
    """Add special-token setters for Transformers initialization and fallback lookups required by ESMC."""

    for token_name in ("cls_token", "eos_token", "mask_token", "pad_token", "bos_token"):
        attr = getattr(tokenizer_cls, token_name, None)
        if isinstance(attr, property) and attr.fset is None:
            def _make_setter(name):
                def _setter(self, value):
                    setattr(self, f"_{name}", value)
                    if not hasattr(self, "_special_tokens_map"):
                        self._special_tokens_map = {}
                    self._special_tokens_map[name] = value
                return _setter
            setattr(
                tokenizer_cls,
                token_name,
                property(attr.fget, _make_setter(token_name), attr.fdel, attr.__doc__),
            )


    def __getattr__(self, name):
        if name in _ESMC_SPECIAL_TOKENS:
            return _ESMC_SPECIAL_TOKENS[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    tokenizer_cls.__getattr__ = __getattr__


_patch_esmc_tokenizer_cls(_get_esmc_tokenizer_cls())


def load_esmc_tokenizer():
    """Load the ESMC sequence tokenizer (patch already applied at import time)."""
    return _get_esmc_tokenizer_cls()()
