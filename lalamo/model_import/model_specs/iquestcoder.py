from lalamo.model_import.model_configs import HFIQuestCoderConfig

from .common import ConfigMap, FileSpec, ModelSpec, UseCase

__all__ = ["IQUESTCODER_MODELS"]


IQUESTCODER_MODELS = [
    ModelSpec(
        vendor="IQuestLab",
        family="IQuest-Coder",
        name="IQuest-Coder-V1-40B-Base-Stage1",
        size="40B",
        quantization=None,
        repo="IQuestLab/IQuest-Coder-V1-40B-Base-Stage1",
        config_type=HFIQuestCoderConfig,
        configs=ConfigMap(tokenizer=FileSpec("tokenizer.model")),
        use_cases=(UseCase.CODE,),
    ),
    ModelSpec(
        vendor="IQuestLab",
        family="IQuest-Coder",
        name="IQuest-Coder-V1-40B-Instruct",
        size="40B",
        quantization=None,
        repo="IQuestLab/IQuest-Coder-V1-40B-Instruct",
        config_type=HFIQuestCoderConfig,
        configs=ConfigMap(tokenizer=FileSpec("tokenizer.model")),
        use_cases=(UseCase.CODE,),
    ),
]
