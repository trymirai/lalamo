import jax
import pytest
import transformers
from jaxtyping import PRNGKeyArray

from fartsovka.importers.huggingface.importer import HuggingFaceModel, import_model
from fartsovka.modules.llama import BaselineLlama

RANDOM_SEED = 42


@pytest.fixture
def rng_key() -> PRNGKeyArray:
    return jax.random.PRNGKey(RANDOM_SEED)


@pytest.fixture(scope="package")
def huggingface_llama() -> transformers.LlamaModel:
    model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model.eval()
    return model


@pytest.fixture(scope="package")
def fartsovka_llama() -> BaselineLlama:
    model = import_model(HuggingFaceModel.LLAMA32_1B_INSTRUCT)
    return model
