import pytest

from lalamo.model_import.model_specs import ModelType
from lalamo.model_import.model_specs.common import ModelSpec
from lalamo.models import ClassifierModelConfig, LanguageModelConfig, TTSGenerator
from tests.conftest import ConvertModel


def test_model_conversion(all_model_specs: ModelSpec, convert_model: ConvertModel) -> None:
    converted_path = convert_model(all_model_specs.repo)

    match all_model_specs.model_type:
        case ModelType.LANGUAGE_MODEL:
            model = LanguageModelConfig.load_model(converted_path)
        case ModelType.CLASSIFIER_MODEL:
            model = ClassifierModelConfig.load_model(converted_path)
        case ModelType.TTS_MODEL:
            model = TTSGenerator.load_model(converted_path)
        case _:
            pytest.fail(f"Unknown model type: {all_model_specs.model_type}")

    del model
