from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tokenizers import AddedToken

from lalamo.model_import.huggingface_tokenizer_config import HFTokenizerConfig

__all__ = ["EncodingLike", "SentencePieceTokenizer", "TokenizerLike"]


class EncodingLike(Protocol):
    ids: list[int]


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> EncodingLike: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str: ...


@dataclass(frozen=True)
class _SimpleEncoding:
    ids: list[int]


class SentencePieceTokenizer:
    """A minimal tokenizer adapter for HuggingFace-style SentencePiece tokenizers.

    This supports:
    - `tokenizer.model` (SentencePiece) via `sentencepiece.SentencePieceProcessor`
    - HF `added_tokens_decoder` (ids >= sp_model.get_piece_size()) by treating them as atomic tokens

    It intentionally implements only the surface area `lalamo` needs:
    `.encode(...).ids` and `.decode(...)`.
    """

    def __init__(self, tokenizer_model_path: Path | str, tokenizer_config: HFTokenizerConfig):
        try:
            import sentencepiece as spm  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "This model ships a SentencePiece `tokenizer.model`. "
                "Install the `sentencepiece` package to convert/run it."
            ) from e

        self._tokenizer_model_path = Path(tokenizer_model_path)
        self._sp = spm.SentencePieceProcessor(**(tokenizer_config.sp_model_kwargs or {}))
        self._sp.Load(str(self._tokenizer_model_path))
        self._sp_piece_size = int(self._sp.get_piece_size())

        self._add_prefix_space = bool(tokenizer_config.add_prefix_space)

        self._added_token_by_id: dict[int, AddedToken] = {}
        self._added_id_by_token: dict[str, int] = {}
        self._special_token_ids: set[int] = set()

        if tokenizer_config.added_tokens_decoder is not None:
            for tok_id_str, hf_token in sorted(
                tokenizer_config.added_tokens_decoder.items(),
                key=lambda kv: int(kv[0]),
            ):
                tok_id = int(tok_id_str)
                token = hf_token.to_added_token()
                self._added_token_by_id[tok_id] = token
                self._added_id_by_token[token.content] = tok_id
                if token.special:
                    self._special_token_ids.add(tok_id)

        # Prefer longest tokens first to avoid partial matches (e.g. "<|im_start|>" vs "<|im_...").
        self._match_tokens = sorted(self._added_id_by_token.keys(), key=len, reverse=True)

    def encode(self, text: str, add_special_tokens: bool = False) -> _SimpleEncoding:  # noqa: ARG002
        if self._add_prefix_space and text and not text.startswith(" "):
            text = " " + text

        ids: list[int] = []
        i = 0
        last = 0
        n = len(text)
        while i < n:
            matched: str | None = None
            for tok in self._match_tokens:
                if text.startswith(tok, i):
                    matched = tok
                    break
            if matched is None:
                i += 1
                continue

            if last < i:
                segment = text[last:i]
                if segment:
                    ids.extend(self._sp.encode(segment, out_type=int))
            ids.append(self._added_id_by_token[matched])
            i += len(matched)
            last = i

        if last < n:
            segment = text[last:]
            if segment:
                ids.extend(self._sp.encode(segment, out_type=int))

        return _SimpleEncoding(ids=ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        out_parts: list[str] = []
        buf: list[int] = []

        def flush_buf() -> None:
            nonlocal buf
            if buf:
                out_parts.append(self._sp.decode(buf))
                buf = []

        for tok_id in ids:
            tok_id_int = int(tok_id)

            # Skip special tokens if requested.
            if skip_special_tokens and tok_id_int in self._special_token_ids:
                continue

            # Treat added tokens and special tokens as literal strings; never feed them to SentencePiece.
            if tok_id_int in self._special_token_ids or tok_id_int >= self._sp_piece_size:
                flush_buf()
                token = self._added_token_by_id.get(tok_id_int)
                if token is None:
                    # Unknown token id outside SentencePiece vocab; preserve information.
                    out_parts.append(f"<tok:{tok_id_int}>")
                else:
                    out_parts.append(token.content)
            else:
                buf.append(tok_id_int)

        flush_buf()
        return "".join(out_parts)


