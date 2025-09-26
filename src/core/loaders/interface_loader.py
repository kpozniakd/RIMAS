from abc import ABC, abstractmethod
from pathlib import Path


class IntefaceLoader(ABC):
    """Inerface for all data loader(from prviders)"""

    @abstractmethod
    def download_dataset(self, dataset: str) -> Path:
        """Loads a dataset and returns the path to the directory with the files"""
        raise NotImplementedError
