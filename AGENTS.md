The rules below are CRITICAL. Each rule should be respected. Rules can only be overridden by the user.

# Workflow rules:
- When given a coding task, AI Assistant carefully and deeply inspects the codebase to understand the context.
- If the Assistant has access to search tools, it uses them to search for documentation and use case examples for external libraries.
- AI Assistant doesn't ask for permission to read the contents of files and directories.
- When given a task, AI assistant first thinks about multiple high-level strategies, and then asks the user for feedback on every option.
- AI Assistant splits tasks into smallest possible subtasks, and solves them one by one while asking the user for detailed feedback during every step.
- AI Assistant never edits code before proposing a short draft to the user and receiving a feedback.
- AI Assistant is not allowed to make even the tiniest architectural decisions, it can only propose multiple options to the user.
- AI Assistant behaves like an interviewer and asks the user questions all the time.
- AI Assistant never relies on its own assumptions about implementation details of libraries and tools. It always makes sure to check the reference.
- Instead of asking multiple questions in one message, AI assistant asks them one by one.
- AI Assistant always asks for permission before launching CLI tools which may take lots of time or produce side effects.
- Before using any function from an external library, AI Assistant thoroughly inspects the code or documentation for the function to ensure that it uses it correctly.
- When encountering an error, AI Assistant never rushes to quickly fix it. Instead, it uses the scientific method to understand the source of the error: comes up with hypotheses and tests them by performing a series of experiments. It only attempts to fix the error once the hypothesis has been sufficiently validated.
- AI Assistant avoids creating .md files with explanations unless explicitly requested by the user.

# General coding guidelines:
- AI Assistant never resorts to quick hacks or stubs to make something work. Instead it always investigates the problem thoroughly and comes up with a well-thought-out solution.
- By default, AI Assistant does not write documentation or comments unless the code contains complex logic which is difficult to understand without them.
- AI Assistant strives for simplicity and maintainability. AI Assistant avoids writing boilerplate code, instead it tries to come up with the right abstractions.
- AI Assistant prefers functional programming style, namely:
  * Avoids mutable state.
  * Avoids modifying objects in-place.
  * Avoids implicit side effects.
  * Prefers immutable data structures.
  * Strives to make invalid state unrepresentable.
- AI Assistant tries to write code in the way that makes bugs hard to make and easy to spot.
- AI Assistant writes the code so that assumptions about invariants are as explicit as possible.
- AI Assistant leverages strong typing as much as possible, for example, it always uses enums instead of literals.
- AI Assistant uses good descriptive variable names that make it easy to understand the purpose of the variable without looking at the code that uses it.
- AI Assistant does not manually parse things such as JSON configs. Instead it first models a schema using types, and then uses a serialisation library such as Cattrs, Pydantic, or Serde to handle serialization and deserialization and validation.

# Testing guidelines
- AI Assistant strives for simplicity and minimalism when writing tests.
- AI Assistant avoids extensive mocking and stubbing.
- AI Assistant focuses on testing core functionality, and does not test minor details such as property accessors.
- AI Assistant avoids boilerplate in the testing code, and tries to come up with the right generic abstractions.

# Python coding rules:
- Never use pip to manage dependencies. Instead use uv commands, such as `uv add`.
- Assume Python 3.12+. Use `T | None` instead of `Optional[T]` and `dict[K, V]` instead of `typing.Dict[K, V]`.
- Every function should have full type annotations.
- Prefer comprehensions to map and filter expressions.
- Prefer dataclasses over vanilla python classes.
- All path-like arguments should have type `Path | str`.
- Prefer frozen dataclasses or named tuples as primary data structures.
- Never use dicts in place of dataclasses.
- __init__.py files should only contain reexports.

# Rust coding rules:
- Prefer `Box<[T]>` to `Vec<T>` if the contents of the container are not going to change.
- Avoid mutable variables as much as possible. Use iterator expressions to collect data into containers.

# Code best practices in examples

```python
# Bad: This expression does not communicate implicit assumptions about the list's shape, and variable names are not descriptive enough.
x, y = my_list[0], my_list[-1]

# Good: Destructuring clearly communicates the assumptions about the length of the list, and variable names are clear.
first, *_, last = my_list
```

```python
# Bad: it is not clear what values the dish can hold.
def cook(dish: str) -> ...:
    ...

# Good: Enum makes it clear.
from enum import Enum

class Dish(Enum):
    STEAK = "steak"
    PIZZA = "pizza"
    SALAD = "salad"

def cook(dish: Dish) -> ...:
    ...
```

```python
# Bad: This expression makes the shape of the `x` and the semantics of the operation unclear.
permuted_x = np.permute(x, (0, 2, 1))

# Good: This expression makes the shape of the `x` and the semantics of the operation clear.
permuted_x = einops.rearrange(x, "batch in_channels out_channels -> batch out_channels in_channels")
```

```python
# Bad: Dict is used instead of a struct.
def model_registry():
    """Registry of all available models"""
    return {
        "llama": {
            "hf_name": "meta-llama/Llama-3.2-1B-Instruct",
            "test_class": LlamaModelTest,
            "max_layers": 16,
        },
        "qwen": {
            "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "test_class": QwenModelTest,
            "max_layers": 24,
        },
    }

# Good: Frozen dataclasses.
@dataclass(frozen=True)
class ModelTestSpec:
    hf_name: str
    test_class: type[BaseModelTest]
    max_layers: int

def model_registry() -> list[ModelTestSpec]:
    return [
        ModelTestSpec(
            hf_name="meta-llama/Llama-3.2-1B-Instruct",
            test_class=LlamaModelTest,
            max_layers=16,
        ),
        ModelTestSpec(
            hf_name="Qwen/Qwen2.5-1.5B-Instruct",
            test_class=QwenModelTest,
            max_layers=24,
        ),
    ]
```

```python
# Bad: The comment is unnecessary and refers to internals of the function
def new(self, key: jax.Array | None = None) -> "BraxAntProblem":
    """Create a new problem instance with a different data key."""
    if key is None:
        key, _ = jr.split(self.data_key)
    return eqx.tree_at(lambda p: p.data_key, self, key)

# Good: No comment, correct type annotation
def new(self, key: Key[Array, ""] | None = None) -> Self:
    if key is None:
        key, _ = jr.split(self.data_key)
    return eqx.tree_at(lambda tree: tree.data_key, self, key)
```

```python
# Bad: Using an unnecessary global constant
NUMBER_OF_SOLVER_STEPS = 5
def solve(self, problem: Problem, num_steps: int = NUMBER_OF_SOLVER_STEPS):
    ...

# Good: In-place constant
def solve(self, problem: Problem, num_steps: int = 5):
    ...
```

```python
# Bad: __init__ method heavily changes dataclass fields
@dataclass
class Banana:
    color: Color
    length: float
    width: float

    def __init__(self, is_green: bool = False):
        if is_green:
            self.color = Color.Green
            self.length = 5.0
            self.width = 3.0
        else:
            self.color = Color.Yellow
            self.length = 7.0
            self.width = 1.0

# Good: A special method to construct the dataclass, boolean argument keyword-only
@dataclass
class Banana:
    color: Color
    length: float
    width: float

    @staticmethod
    def build(cls, *, is_green: bool):
        ...
```

```python
# Bad: Mixing semantic and operational arguments together in one structure
@dataclass
class Config:
    vram_limit: int # semantic
    use_flash_attention: bool # operational
    ...

# Good: Split them up
@dataclass
class ParallelizationConfig:
    vram_limit: int
    ...

@dataclass
class ForwardPassConfig:
    use_flash_attention: bool
    ...
```

```python
# Bad: An unnecessary function and confusing logic
def _apply_sharding_config(array: Float[Array, "*"], sharding: Sharding | None):
    if sharding is None:
        return array
    return sharding.shard(array)

def shard_module(module: ShardableModule):
    for field in module.fields():
        if eqx.is_array(field):
            field = _apply_sharding_config(field, module.get_sharding())
        ...

# Good: No unnecessary functions, easy to read and understand
def shard_module(module: ShardableModule):
    sharding = module.get_sharding()

    for field in module.fields():
        if eqx.is_array(field) and sharding:
            field = sharding.shard(field)
        ...
```

```python
# Bad: message is non informative and contains trivial details
def test_decode_one_token():
    skip_on_gpu("Skip on GPU due to token mismatch in test_decode_one_token.")
    ...

# Good: message describes the reasoning
def test_decode_one_token():
    skip_on_gpu("Flaky on GPU due to torch/jax precision inconsistencies.")
    ...
```

Some well-written code of mine:

```python
def load_tokenized_dataset(
    message_processor,
    dataset_name: str,
    data_dir: Path | str = "data",
    subsample_size: int | None = None,
) -> list[list[int]]:
    data_dir = Path(data_dir)
    dataset_path = data_dir / f"{dataset_name.replace('/', '_')}_processed.txt"
    if dataset_path.exists():
        result = []
        with open(dataset_path) as file:
            for i, line in enumerate(file):
                if subsample_size is not None and i >= subsample_size:
                    break
                result.append(list(map(int, line.split())))
            return result

    raw_dataset = datasets.load_dataset(dataset_name, split="train").to_polars()[
        "conversation"
    ]
    result = []
    for chat in tqdm(raw_dataset):
        processed_chat = [
            llm.data.huggingface_message.HFMessage.from_dict(message).as_message()
            for message in chat
        ]
        result.append(message_processor.tokenize_request(processed_chat))

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w") as file:
        for token_ids in result:
            print(*map(str, token_ids), file=file)

    return result

def map_modules(
    module_type: type[eqx.Module],
    map_fn: Callable[[eqx.Module], eqx.Module],
    model: eqx.Module,
) -> eqx.Module:
    def wrapper(module: eqx.Module) -> eqx.Module:
        if isinstance(module, module_type):
            return map_fn(module)
        return module

    return jax.tree.map(wrapper, model, is_leaf=lambda x: isinstance(x, module_type))

@eqx.filter_jit
def kl_divergence(
    p_model: llm.LanguageModel, q_model: llm.LanguageModel, batch: InputBatch
) -> Float[Array, "batch"]:
    _, sequence_length = batch.token_ids.shape
    p_logits = compute_logits(p_model, batch)
    q_logits = compute_logits(q_model, batch)
    unmasked_kl_div = jnp.sum(jnp.exp(p_logits) * (p_logits - q_logits), axis=-1)
    mask = jnp.arrange(sequence_length)[None, :] < batch.sequence_lengths[:, None]
    masked_kl_div = jnp.where(mask, unmasked_kl_div, 0.0)
    return jnp.sum(masked_kl_div) / jnp.sum(batch.sequence_lengths)

def evaluate_model(
    reference_model: llm.LanguageModel,
    model: llm.LanguageModel,
    dataset: list[list[int]],
    batch_size: int = 32,
    sequence_length: int = 1024,
) -> float:
    kl_divs = []
    for batch_token_ids in tqdm(
        batched(dataset, batch_size),
        total=math.ceil(len(dataset) / batch_size),
    ):
        batch = InputBatch.from_tokens(
            list(batch_token_ids), sequence_length=sequence_length
        )
        kl_div = kl_divergence(reference_model, model, batch).item()
        kl_divs.append(kl_div)

    return sum(kl_divs) / len(kl_divs)
```
