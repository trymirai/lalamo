import equinox as eqx
import huggingface_hub
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from lalamo.data import load_hf_parquet
from lalamo.data.huggingface_message import HFMessage
from lalamo.message_processor import AssistantMessage, Message, MessageProcessor, UserMessage

LMSYS_PARQUET_PATH = "data/train-00000-of-00006-4feeb3f83346a0e9.parquet"


class Batch(eqx.Module):
    token_ids: Int[Array, "batch seq_len"]
    positions: Int[Array, "batch seq_len"]
    loss_mask: Bool[Array, "batch seq_len"]


def load_lmsys_calibration_texts(num_sequences: int = 32) -> list[str]:
    raw_conversations = _load_raw_conversations(num_sequences * 10)

    texts: list[str] = []
    for conversation in raw_conversations:
        messages = [HFMessage.from_dict(turn) for turn in conversation]
        first_user = next((m for m in messages if m.role in ("user", "human")), None)
        if first_user is not None and len(first_user.content) > 20:
            texts.append(first_user.content)
            if len(texts) >= num_sequences:
                break
    return texts


def load_lmsys_conversations(num_sequences: int = 100) -> list[list[Message]]:
    raw_conversations = _load_raw_conversations(num_sequences * 10)

    conversations: list[list[Message]] = []
    for raw in raw_conversations:
        messages = [HFMessage.from_dict(turn).as_message() for turn in raw]
        if any(isinstance(m, AssistantMessage) for m in messages):
            conversations.append(messages)
            if len(conversations) >= num_sequences:
                break
    return conversations


def _load_raw_conversations(num_rows: int) -> list:
    local_path = huggingface_hub.hf_hub_download(
        "lmsys/lmsys-chat-1m",
        LMSYS_PARQUET_PATH,
        repo_type="dataset",
    )
    return load_hf_parquet(local_path).head(num_rows).collect().get_column("conversation").to_list()


def _compute_gen_prompt_len(tokenizer: MessageProcessor) -> int:
    request = tokenizer.request_to_dict([UserMessage("test")])
    with_prompt = tokenizer.prompt_template.render({**request, "strftime_now": lambda _: ""})
    without_prompt = tokenizer.prompt_template.render(
        {**request, "add_generation_prompt": False, "strftime_now": lambda _: ""}
    )
    return len(tokenizer.tokenize_text(with_prompt[len(without_prompt) :]))


def make_batch(conversations: list[list[Message]], tokenizer: MessageProcessor, seq_len: int = 256) -> Batch:
    gen_prompt_len = _compute_gen_prompt_len(tokenizer)

    all_token_ids: list[list[int]] = []
    all_masks: list[list[bool]] = []

    for messages in conversations:
        full_tokens = tokenizer.tokenize_request(messages)[:-gen_prompt_len][:seq_len]
        mask = [False] * len(full_tokens)

        for i, msg in enumerate(messages):
            if isinstance(msg, AssistantMessage):
                start = len(tokenizer.tokenize_request(messages[:i]))
                end = len(tokenizer.tokenize_request(messages[: i + 1])) - gen_prompt_len
                for j in range(start, min(end, len(full_tokens))):
                    mask[j] = True

        padding = seq_len - len(full_tokens)
        all_token_ids.append(full_tokens + [0] * padding)
        all_masks.append(mask + [False] * padding)

    token_ids = jnp.array(all_token_ids, dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), token_ids.shape)
    loss_mask = jnp.array(all_masks, dtype=jnp.bool_)
    return Batch(token_ids=token_ids, positions=positions, loss_mask=loss_mask)


def tokenize_batch(
    texts: list[str], tokenizer: MessageProcessor, seq_len: int = 256
) -> tuple[Int[Array, "batch seq_len"], Int[Array, "batch seq_len"]]:
    token_lists = [tokenizer.tokenize_text(text)[:seq_len] for text in texts]
    padded = jnp.array([tokens + [0] * (seq_len - len(tokens)) for tokens in token_lists], dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), padded.shape)
    return padded, positions


def kl_divergence(
    teacher_logits: Float[Array, "batch seq_len vocab"],
    student_logits: Float[Array, "batch seq_len vocab"],
    mask: Bool[Array, "batch seq_len"] | None = None,
) -> Float[Array, ""]:
    teacher_probs = jax.nn.softmax(teacher_logits)
    kl_per_position = (teacher_probs * (jax.nn.log_softmax(teacher_logits) - jax.nn.log_softmax(student_logits))).sum(
        -1
    )
    if mask is None:
        return kl_per_position.mean()
    return (kl_per_position * mask).sum() / mask.sum()


def eval_kl(
    teacher_decoder: eqx.Module,
    student_decoder: eqx.Module,
    token_ids: Int[Array, "batch seq_len"],
    positions: Int[Array, "batch seq_len"],
) -> float:
    teacher_logits = teacher_decoder(token_ids, positions).logits
    student_logits = student_decoder(token_ids, positions).logits
    return float(kl_divergence(teacher_logits, student_logits))
