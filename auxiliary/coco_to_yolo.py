# src/coco_to_yolo.py
# конвертация + валидация
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict


def load_coco(path: str | Path) -> Dict:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def build_image_index(coco: Dict) -> Dict[int, Dict]:
    return {img["id"]: img for img in coco["images"]}


def group_annotations_by_image(coco: Dict) -> Dict[int, List[Dict]]:
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
    return by_image


def get_fish_category_id(coco: Dict, fish_name: str = "fish") -> int:
    for cat in coco["categories"]:
        if cat["name"] == fish_name:
            return cat["id"]
    raise ValueError(f"Category '{fish_name}' not found in COCO categories")


def convert_bbox_coco_to_yolo(bbox_xywh: List[float], img_w: int, img_h: int) -> List[float]:
    x_min, y_min, w, h = bbox_xywh
    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0
    return [
        x_center / img_w,
        y_center / img_h,
        w / img_w,
        h / img_h,
    ]


def convert_coco_to_yolo_for_split(
    images_dir: str | Path,
    annotations_path: str | Path,
    labels_out_dir: str | Path,
    class_name: str = "fish",
    yolo_class_id: int = 0,
) -> Dict[str, int]:
    """
    Конвертирует все боксы класса fish из COCO в YOLO формат.
    Возвращает статистику:
      total_images, images_with_fish, total_fish_boxes, missing_images
    """
    images_dir = Path(images_dir)
    annotations_path = Path(annotations_path)
    labels_out_dir = Path(labels_out_dir)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco(annotations_path)
    img_index = build_image_index(coco)
    anns_by_img = group_annotations_by_image(coco)
    fish_cat_id = get_fish_category_id(coco, class_name)

    for old_txt in labels_out_dir.glob("*.txt"):
        old_txt.unlink()

    total_images = len(img_index)
    images_with_fish = 0
    total_fish_boxes = 0
    missing_images = 0

    for img_id, img_info in img_index.items():
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        img_path = images_dir / file_name
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            missing_images += 1
            continue

        anns = anns_by_img.get(img_id, [])
        yolo_lines: List[str] = []
        fish_boxes_in_image = 0

        for ann in anns:
            if ann["category_id"] != fish_cat_id:
                continue

            bbox = ann["bbox"]
            x_center, y_center, w_norm, h_norm = convert_bbox_coco_to_yolo(
                bbox, img_w, img_h
            )
            line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_lines.append(line)
            fish_boxes_in_image += 1

        if fish_boxes_in_image > 0:
            images_with_fish += 1
            total_fish_boxes += fish_boxes_in_image
            label_path = labels_out_dir / (Path(file_name).stem + ".txt")
            with label_path.open("w") as f:
                f.write("\n".join(yolo_lines))

    return {
        "total_images": total_images,
        "images_with_fish": images_with_fish,
        "total_fish_boxes": total_fish_boxes,
        "missing_images": missing_images,
    }
