from __future__ import annotations

import os
import tempfile
from typing import Optional

from pdf2image import convert_from_path

from core.file_converters.file_converter import FileConverter


class PDFConverter(FileConverter):
    def convert_to_images(self, file_path: str):
        return convert_from_path(file_path)

    def convert_and_save_images(self, file_path: str, output_folder: str | None = None):
        images = self.convert_to_images(file_path)
        if not output_folder:
            # set tmp folder as output folder
            output_folder = tempfile.gettempdir()
        os.makedirs(output_folder, exist_ok=True)
        # save images to output folder
        output_file_paths = []
        for i, image in enumerate(images):
            output_file_path = os.path.join(output_folder, f"page_{i}.png")
            image.save(output_file_path)
            output_file_paths.append(output_file_path)
        return output_file_paths
