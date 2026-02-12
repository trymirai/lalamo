from dataclasses import dataclass
from pathlib import Path


@dataclass
class LalamoEngineConfig:
    model_path: Path
    batch_size: int | None = None
    vram_gb: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size is not None and self.vram_gb is not None:
            raise ValueError("Cannot specify both batch_size and vram_gb")

    def get_model_name(self) -> str:
        return self.model_path.name
