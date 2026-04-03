from lalamo.model_import.model_configs.huggingface.llama import HFLlamaConfig
from lalamo.model_import.model_specs.common import ConfigMap, FileSpec, JSONFieldSpec, ModelSpec

DIRECT_TEMPLATE = "{% for message in messages %}{{ message.content }}{% endfor %}"


class TestModelSpecWithChatTemplate:
    def test_model_spec_with_string_chat_template(self) -> None:
        spec = ModelSpec(
            vendor="Test",
            family="Test",
            name="Test",
            size="1B",
            repo="test/test",
            config_type=HFLlamaConfig,
            configs=ConfigMap(chat_template=DIRECT_TEMPLATE),
        )
        assert spec.configs.chat_template == DIRECT_TEMPLATE

    def test_model_spec_with_file_spec_chat_template(self) -> None:
        spec = ModelSpec(
            vendor="Test",
            family="Test",
            name="Test",
            size="1B",
            repo="test/test",
            config_type=HFLlamaConfig,
            configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
        )
        assert isinstance(spec.configs.chat_template, FileSpec)
        assert spec.configs.chat_template.filename == "chat_template.jinja"

    def test_model_spec_with_json_field_spec_chat_template(self) -> None:
        spec = ModelSpec(
            vendor="Test",
            family="Test",
            name="Test",
            size="1B",
            repo="test/test",
            config_type=HFLlamaConfig,
            configs=ConfigMap(chat_template=JSONFieldSpec(FileSpec("tokenizer_config.json"), "chat_template")),
        )
        assert isinstance(spec.configs.chat_template, JSONFieldSpec)
        assert spec.configs.chat_template.field_name == "chat_template"


class TestModelSpecJsonSerialization:
    def test_roundtrip_with_string_chat_template(self) -> None:
        spec = ModelSpec(
            vendor="Test",
            family="Test",
            name="Test",
            size="1B",
            repo="test/test",
            config_type=HFLlamaConfig,
            configs=ConfigMap(chat_template=DIRECT_TEMPLATE),
        )
        json_data = spec.to_json()
        restored = ModelSpec.from_json(json_data)
        assert restored.configs.chat_template == DIRECT_TEMPLATE

    def test_roundtrip_with_file_spec_chat_template(self) -> None:
        spec = ModelSpec(
            vendor="Test",
            family="Test",
            name="Test",
            size="1B",
            repo="test/test",
            config_type=HFLlamaConfig,
            configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
        )
        json_data = spec.to_json()
        restored = ModelSpec.from_json(json_data)
        assert isinstance(restored.configs.chat_template, FileSpec)
        assert restored.configs.chat_template.filename == "chat_template.jinja"

    def test_roundtrip_with_json_field_spec_chat_template(self) -> None:
        spec = ModelSpec(
            vendor="Test",
            family="Test",
            name="Test",
            size="1B",
            repo="test/test",
            config_type=HFLlamaConfig,
            configs=ConfigMap(chat_template=JSONFieldSpec(FileSpec("config.json"), "chat_template")),
        )
        json_data = spec.to_json()
        restored = ModelSpec.from_json(json_data)
        assert isinstance(restored.configs.chat_template, JSONFieldSpec)
        assert restored.configs.chat_template.field_name == "chat_template"
        assert restored.configs.chat_template.file_spec.filename == "config.json"

    def test_from_json_with_string_chat_template(self) -> None:
        json_data = {
            "vendor": "Test",
            "family": "Test",
            "name": "Test",
            "size": "1B",
            "repo": "test/test",
            "config_type": "HFLlamaConfig",
            "configs": {
                "chat_template": DIRECT_TEMPLATE,
            },
        }
        spec = ModelSpec.from_json(json_data)
        assert spec.configs.chat_template == DIRECT_TEMPLATE

    def test_from_json_with_file_spec_chat_template(self) -> None:
        json_data = {
            "vendor": "Test",
            "family": "Test",
            "name": "Test",
            "size": "1B",
            "repo": "test/test",
            "config_type": "HFLlamaConfig",
            "configs": {
                "chat_template": {"filename": "chat_template.jinja"},
            },
        }
        spec = ModelSpec.from_json(json_data)
        assert isinstance(spec.configs.chat_template, FileSpec)
        assert spec.configs.chat_template.filename == "chat_template.jinja"

    def test_from_json_with_json_field_spec_chat_template(self) -> None:
        json_data = {
            "vendor": "Test",
            "family": "Test",
            "name": "Test",
            "size": "1B",
            "repo": "test/test",
            "config_type": "HFLlamaConfig",
            "configs": {
                "chat_template": {
                    "file_spec": {"filename": "config.json"},
                    "field_name": "chat_template",
                },
            },
        }
        spec = ModelSpec.from_json(json_data)
        assert isinstance(spec.configs.chat_template, JSONFieldSpec)
        assert spec.configs.chat_template.field_name == "chat_template"
