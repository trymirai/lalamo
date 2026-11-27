import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, NewType

import cattrs
from tokenizers import AddedToken

__all__ = ["HFAddedToken", "HFTokenizerConfig"]

SpecialToken = NewType("SpecialToken", str)


@dataclass(frozen=True)
class HFAddedToken:
    content: str
    single_word: bool
    normalized: bool
    special: bool
    lstrip: bool
    rstrip: bool

    def to_added_token(self) -> AddedToken:
        return AddedToken(
            self.content,
            single_word=self.single_word,
            normalized=self.normalized,
            special=self.special,
            lstrip=self.lstrip,
            rstrip=self.rstrip,
        )


@dataclass(frozen=True)
class HFSerializedSpecialToken:
    content: str


def _structure_special_token(value: str | dict[str, object], _: object) -> SpecialToken:
    # Older tokenizer configs store special tokens as serialized AddedToken objects instead of plain strings.
    if isinstance(value, str):
        return SpecialToken(value)
    return SpecialToken(cattrs.structure(value, HFSerializedSpecialToken).content)


@dataclass(frozen=True)
class HFTokenizerConfig:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(int | list[int], lambda v, _: v)
    _converter.register_structure_hook(SpecialToken, _structure_special_token)

    # ---------- core identity ----------
    tokenizer_class: str | None = None
    model_max_length: int | None = None
    padding_side: str | None = None  # "left" | "right"
    truncation_side: str | None = None  # "left" | "right"
    legacy: bool | None = None
    use_fast: bool | None = None
    clean_up_tokenization_spaces: bool | None = None

    # ---------- behaviour flags ----------
    add_bos_token: bool | None = None
    add_eos_token: bool | None = None
    add_prefix_space: bool | None = None
    use_default_system_prompt: bool | None = None
    spaces_between_special_tokens: bool | None = None
    do_lower_case: bool | None = None

    # ---------- special tokens ----------
    bos_token: SpecialToken | None = None
    eos_token: SpecialToken | None = None
    unk_token: SpecialToken | None = None
    pad_token: SpecialToken | None = None
    sep_token: SpecialToken | None = None
    cls_token: SpecialToken | None = None
    mask_token: SpecialToken | None = None
    added_tokens_decoder: dict[str, HFAddedToken] | None = None

    # ---------- chat / SentencePiece ----------
    chat_template: str | None = None
    sp_model_kwargs: dict | None = None

    # ---------- extras ----------
    language: str | None = None
    task: str | None = None

    def added_tokens(self) -> list[AddedToken]:
        if self.added_tokens_decoder is None:
            return []
        return [token.to_added_token() for token in self.added_tokens_decoder.values()]

    @classmethod
    def from_json(cls, json_path: Path | str) -> "HFTokenizerConfig":
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        return cls._converter.structure(config, cls)
