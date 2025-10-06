from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class FileConverter(ABC):
    @abstractmethod
    def convert_to_images(self, file_path: str):
        pass
