"""Compatibility helpers for EvolutionaryScale ESMC tokenization."""

# ESM sequence vocabulary: ["<cls>", "<pad>", "<eos>", "<unk>", ...]
# Special token strings are fixed across all ESMC releases.
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
    """Patch EsmSequenceTokenizer for compatibility with HuggingFace and ESMC internals.

    Two problems fixed:

    1. Some ESM releases make special-token properties read-only, but newer
       transformers tries to set them during tokenizer __init__. Fixed by
       adding a setter.

    2. EsmSequenceTokenizer._get_token(name) falls back to
       self.__getattr__(name) when _special_tokens_map doesn't contain the
       key, but the class never defines __getattr__, causing AttributeError.
       This crashes both HuggingFace padding machinery AND the ESMC model's
       own forward() which calls self.tokenizer.pad_token_id.
       Fixed by adding __getattr__ that returns the known token strings.
    """
    # Fix 1: add setters for read-only special-token properties
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

    # Fix 2: add __getattr__ fallback for special token name lookups.
    # Applied unconditionally — it only triggers when normal lookup fails,
    # so it is safe to add even if the class later defines its own __getattr__.
    def __getattr__(self, name):
        if name in _ESMC_SPECIAL_TOKENS:
            return _ESMC_SPECIAL_TOKENS[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    tokenizer_cls.__getattr__ = __getattr__


# Patch at import time so the fix applies to ALL EsmSequenceTokenizer instances,
# including the one embedded inside the ESMC model checkpoint.
_patch_esmc_tokenizer_cls(_get_esmc_tokenizer_cls())


def load_esmc_tokenizer():
    """Load the ESMC sequence tokenizer (patch already applied at import time)."""
    return _get_esmc_tokenizer_cls()()
