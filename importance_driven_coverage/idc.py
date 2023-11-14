from typing import Tuple

from torch import nn
from torch.utils.data import DataLoader


class ImportanceDrivenCoverage:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def setup(self, data: DataLoader) -> None:
        pass

    def calculate(self, data: DataLoader) -> Tuple[float, set]:
        return (0.0, set())
