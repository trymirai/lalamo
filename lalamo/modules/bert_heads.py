from dataclasses import dataclass

from .common import LalamoModule

@dataclass(frozen=True)
class ModernBertPredictionHeadConfig():
    pass

class ModernBertPredictionHead(LalamoModule[ModernBertPredictionHeadConfig]):
    pass
