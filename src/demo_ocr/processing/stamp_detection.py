import os
from pathlib import Path

import cv2
import fitz
import numpy as np
from ultralytics import YOLO

class StampDetection:
    
    def __init__(self, model_path: str, confidence: float = 0.2):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self._results = None
        self._images = []

    def detect(self, pdf_path: str, dpi: int = 200, show: bool = True) -> dict:
        images = self._pdf_to_images(pdf_path, dpi)
        self._images = []
        details = []
        pages_with_stamp = 0

        for page_num, img in enumerate(images, start=1):
            result = self.model.predict(img, conf=self.confidence, verbose=False)[0]
            boxes = result.boxes

            page_result = {
                "page": page_num,
                "stamp_found": len(boxes) > 0,
                "stamp_count": len(boxes),
                "bboxes": [],
                "confidences": []
            }

            if len(boxes) > 0:
                pages_with_stamp += 1
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    page_result["bboxes"].append([round(x, 2) for x in xyxy])
                    page_result["confidences"].append(round(conf, 4))

                if show:
                    img_with_boxes = self._draw_boxes(img.copy(), page_result)
                    self._images.append((page_num, img_with_boxes))

                    tmp_dir = "/data/result_ocr/tmp_images"
                    os.makedirs(tmp_dir, exist_ok=True)
                    save_name = Path(pdf_path).stem + ".png"
                    cv2.imwrite(os.path.join(tmp_dir, save_name), cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))

            details.append(page_result)

            break

        self._results = {
            "file": Path(pdf_path).name,
            "total_pages": len(images),
            "pages_with_stamp": pages_with_stamp,
            "confidence_threshold": self.confidence,
            "details": details
        }

        if show:
            self.save_images()

        return self._results

    def _pdf_to_images(self, pdf_path: str, dpi: int) -> list:
        doc = fitz.open(pdf_path)
        images = []
        matrix = fitz.Matrix(dpi / 72, dpi / 72)

        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            images.append(img)

        doc.close()
        return images

    def _draw_boxes(self, img: np.ndarray, page_result: dict) -> np.ndarray:
        for bbox, conf in zip(page_result["bboxes"], page_result["confidences"]):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                img, f"stamp {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )
        return img

    def _print_summary(self):
        r = self._results
        print(f"\n{'=' * 50}")
        print(f"File: {r['file']}")
        print(f"{'=' * 50}")
        print(f"Total pages: {r['total_pages']}")
        print(f"Pages with stamp: {r['pages_with_stamp']}")
        print(f"Confidence threshold: {r['confidence_threshold']}")
        print(f"{'-' * 50}")

        for d in r["details"]:
            status = "+" if d["stamp_found"] else "-"
            print(f"[{status}] Page {d['page']}: {d['stamp_count']} stamp(s)")
            for i, conf in enumerate(d["confidences"]):
                print(f"    stamp {i + 1}: {conf:.2%}")

        print(f"{'=' * 50}\n")

    def save_images(self, output_dir: str = "output_images"):
        if not self._images:
            return

        os.makedirs(output_dir, exist_ok=True)
        saved = []

        for page_num, img in self._images:
            filepath = f"{output_dir}/page_{page_num}.jpg"
            cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            saved.append(filepath)

        print(f"Images saved: {output_dir}/")
        for f in saved:
            print(f"  - {Path(f).name}")

