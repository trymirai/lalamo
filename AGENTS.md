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
- AI Assistant inlines trivial helpers/properties and keeps simple implementation details at the use site.
- AI Assistant prefers functional programming style, namely:
  * Avoids mutable state.
  * Avoids modifying objects in-place.
  * Avoids implicit side effects.
  * Prefers immutable data structures.
  * Strives to make invalid state unrepresentable.
- AI Assistant tries to write code in the way that makes bugs hard to make and easy to spot.
- AI Assistant writes the code so that assumptions about invariants are as explicit as possible.
- AI Assistant uses user-facing exceptions for user-actionable errors; use `assert`s for broken configs, unsupported conversion internals, and checkpoint-layout invariants.
- Hot-path runtime checks:
  * Do not use `eqx.error_if` in JIT/model forward/kernel paths unless the path is cold or explicitly debug-only.
  * Treat `eqx.error_if` as a performance hazard, not a harmless assert.
  * Prefer making invalid state impossible before entering the compiled path.
  * For Python-side construction/config validation, use normal `assert` or `ValueError`.
  * For compiled hot paths, avoid dynamic validation unless correctness absolutely requires it and the cost is measured and accepted.
- AI Assistant leverages strong typing as much as possible, for example, it always uses enums instead of literals.
- AI Assistant uses good descriptive variable names that make it easy to understand the purpose of the variable without looking at the code that uses it.
- AI Assistant does not manually parse things such as JSON configs. Instead it first models a schema using types, and then uses a serialisation library such as Cattrs, Pydantic, or Serde to handle serialization and deserialization and validation.
- When adapting checkpoint weights, prefer lazy mapping/patching.

# Testing guidelines
- AI Assistant strives for simplicity and minimalism when writing tests.
- AI Assistant avoids extensive mocking and stubbing.
- AI Assistant focuses on testing core functionality, and does not test minor details such as property accessors.
- AI Assistant avoids tests whose only purpose is to preserve a trivial helper/property.
- AI Assistant should run the tests via `uv run pytest -m fast -n=4` after finishing any non-trivial changes.
- AI Assistant runs pyrefly and ruff checks after every edit to make sure there are no type errors and linter warnings.

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
- Avoid single-line if expressions. Prefer multi-line if statements.

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
# Bad: Using an unnecessary global constant
NUMBER_OF_SOLVER_STEPS = 5
def solve(self, problem: Problem, num_steps: int = NUMBER_OF_SOLVER_STEPS):
    ...

# Good: In-place constant
def solve(self, problem: Problem, num_steps: int = 5):
    ...
```

```python
# Bad: custom __init__ method on a dataclass
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
    def init(cls, *, is_green: bool):
        ...
```

```python
# Bad: Mixing semantic and operational arguments together in one structure
@dataclass
class Config:
    vram_limit: str # semantic
    use_flash_attention: bool # operational
    ...

# Good: Split them up
@dataclass
class ParallelizationConfig:
    vram_limit: str
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

```python
# Bad: a one-use helper hides the actual shape operation.
def _prepare_kv_projection(
    self,
    projection: Float[Array, "tokens channels"],
) -> Float[Array, "tokens groups head_channels"]:
    return rearrange(
        projection,
        "tokens (groups head_channels) -> tokens groups head_channels",
        groups=self.config.num_groups,
        head_channels=self.config.head_dim,
    )

keys = self._prepare_kv_projection(raw_keys)

# Good: the implementation detail is visible where it matters.
keys = rearrange(
    raw_keys,
    "tokens (groups head_channels) -> tokens groups head_channels",
    groups=self.config.num_groups,
    head_channels=self.config.head_dim,
)
```

```python
# Bad: a trivial property exists only to name a simple formula.
@property
def qkv_output_dims(self) -> tuple[int, ...]:
    query_dim = self.num_heads * self.head_dim
    kv_dim = self.num_groups * self.head_dim
    return self.projection_mode.qkv_output_dims(query_dim, kv_dim)

qkv_projection = self.qkv_projection_config.init(
    initializer,
    input_dim=model_dim,
    output_dims=self.qkv_output_dims,
)

# Good: the projection-mode shape is explicit at the construction site.
query_dim = self.num_heads * self.head_dim
kv_dim = self.num_groups * self.head_dim
if self.projection_mode is AttentionProjectionMode.QKV:
    qkv_output_dims = (query_dim, kv_dim, kv_dim)
elif self.projection_mode is AttentionProjectionMode.KEY_SAME_AS_VALUE:
    qkv_output_dims = (query_dim, kv_dim)
else:
    qkv_output_dims = (query_dim,)

qkv_projection = self.qkv_projection_config.init(
    initializer,
    input_dim=model_dim,
    output_dims=qkv_output_dims,
)
```

```python
# Bad: friendly errors for impossible internal checkpoint/config states.
def _validate_gate_up_projection(weights: Array) -> None:
    if weights.ndim != 3:
        raise ValueError(f"Expected batched gate/up projection weights, got shape {weights.shape}.")
    _, gate_up_dim, _ = weights.shape
    if gate_up_dim % 2 != 0:
        raise ValueError(f"Gemma 4 gate/up projection has odd hidden dim {weights.shape}.")

_validate_gate_up_projection(routed_gate_up)

# Good: concise assertions state the developer invariant at the use site.
assert routed_gate_up.ndim == 3
_, gate_up_dim, _ = routed_gate_up.shape
assert gate_up_dim % 2 == 0
```
