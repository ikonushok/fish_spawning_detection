# src/build_yolo_dataset_from_coco.py
import json
import shutil
import logging
from pathlib import Path
from collections import defaultdict

from auxiliary.config_loader import load_config  # как у тебя в проекте


# === НАСТРОЙКИ ===

# имя папки внутри dataset/, где будет YOLO-датасет
YOLO_DATASET_DIRNAME = "yolo_dataset"

# Копировать или перемещать изображения
MOVE_IMAGES = True  # True = shutil.move, False = shutil.copy2


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def coco_to_yolo_split(
    annotations_path: Path,
    src_images_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    class_name: str,
    yolo_class_id: int = 0,
    move_images: bool = False,
):
    """
    Преобразует COCO-аннотации в YOLO txt и переносит/копирует изображения
    в новую структуру.

    COCO:
      images: [{id, file_name, width, height}, ...]
      annotations: [{image_id, bbox, category_id}, ...]
      categories: [{id, name}, ...]

    YOLO txt формат:
      <class_id> <xc> <yc> <w> <h>   (нормировано в [0,1])
    """
    logging.info(f"Читаем COCO-аннотации: {annotations_path}")
    with annotations_path.open("r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    # image_id -> [annotations]
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # mapping category_id -> name
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    # категории, которые считаем целевыми (обычно одна: fish)
    target_cat_ids = {
        cid for cid, name in cat_id_to_name.items() if name == class_name
    }
    if not target_cat_ids:
        logging.warning(
            f"В COCO не найдено категорий с именем '{class_name}'. "
            "Все боксы будут проигнорированы."
        )

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    images_with_boxes = 0
    total_boxes = 0
    missing_images = 0

    for img_id, img_info in images.items():
        total_images += 1

        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        src_img_path = src_images_dir / file_name
        if not src_img_path.exists():
            logging.warning(f"Картинка не найдена: {src_img_path}")
            missing_images += 1
            continue

        # Куда кладём картинку
        dst_img_path = dst_images_dir / file_name
        dst_img_path.parent.mkdir(parents=True, exist_ok=True)

        if move_images:
            shutil.move(str(src_img_path), str(dst_img_path))
        else:
            shutil.copy2(str(src_img_path), str(dst_img_path))

        # Аннотации для этой картинки
        anns = anns_by_image.get(img_id, [])
        yolo_lines = []

        for ann in anns:
            if ann["category_id"] not in target_cat_ids:
                continue

            x_min, y_min, w, h = ann["bbox"]  # COCO: [x,y,width,height] в пикселях

            xc = (x_min + w / 2.0) / width
            yc = (y_min + h / 2.0) / height
            ww = w / width
            hh = h / height

            # Ограничим значения, чтобы не вылазили случайно за [0,1]
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            ww = min(max(ww, 0.0), 1.0)
            hh = min(max(hh, 0.0), 1.0)

            yolo_lines.append(f"{yolo_class_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            total_boxes += 1

        if yolo_lines:
            images_with_boxes += 1
            label_path = dst_labels_dir / (Path(file_name).stem + ".txt")
            with label_path.open("w") as f:
                f.write("\n".join(yolo_lines))
        # если боксов нет — файл .txt не создаём

    logging.info(
        f"Всего картинок: {total_images}, "
        f"с боксами: {images_with_boxes}, "
        f"всего боксов: {total_boxes}, "
        f"отсутствующих картинок: {missing_images}"
    )


def copy_unlabeled_images(
    src_images_dir: Path,
    dst_images_dir: Path,
    move_images: bool = False,
):
    """
    Копирует/перемещает НЕразмеченные кадры в images/Unlabeled.
    Метки не создаём.
    """
    if not src_images_dir.exists():
        logging.warning(f"Папка с unlabeled-картинками не найдена: {src_images_dir}")
        return

    dst_images_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    count = 0

    for img_path in src_images_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in exts:
            continue

        rel_path = img_path.relative_to(src_images_dir)
        dst_path = dst_images_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if move_images:
            shutil.move(str(img_path), str(dst_path))
        else:
            shutil.copy2(str(img_path), str(dst_path))

        count += 1

    logging.info(
        f"Unlabeled: скопировано/перемещено {count} изображений "
        f"из {src_images_dir} в {dst_images_dir}"
    )


def build_yolo_dataset(cfg_path: str = "../configs/config.yaml"):
    setup_logging()
    logging.info(f"Загружаю конфиг: {cfg_path}")

    cfg = load_config(cfg_path)

    # Корень проекта: /Users/.../PycharmProjects/fish_spawning_detection
    project_root = Path(__file__).resolve().parents[1]

    # dataset.root_dir из конфига (относительно корня проекта)
    root_dir_cfg = Path(cfg["dataset"]["root_dir"])
    if root_dir_cfg.is_absolute():
        dataset_root = root_dir_cfg
    else:
        dataset_root = (project_root / root_dir_cfg).resolve()

    logging.info(f"Корень COCO-датасетов (dataset_root): {dataset_root}")

    train_cfg = cfg["dataset"]["train"]
    val_cfg = cfg["dataset"]["val"]
    class_name = cfg["classes"][0]  # предполагаем один класс, например 'fish'

    # Старые пути (COCO)
    train_images_src = (dataset_root / train_cfg["images_dir"]).resolve()
    train_ann_path = (dataset_root / train_cfg["annotations_file"]).resolve()

    val_images_src = (dataset_root / val_cfg["images_dir"]).resolve()
    val_ann_path = (dataset_root / val_cfg["annotations_file"]).resolve()

    # unlabeled (опционально)
    unlabeled_cfg = cfg["dataset"].get("unlabeled", None)
    unlabeled_images_src = None
    if unlabeled_cfg is not None and "images_dir" in unlabeled_cfg:
        unlabeled_images_src = (dataset_root / unlabeled_cfg["images_dir"]).resolve()

    # НОВЫЙ корень YOLO-датасета:
    # /Users/.../fish_spawning_detection/dataset/yolo_dataset
    new_root = dataset_root / YOLO_DATASET_DIRNAME

    # Структура:
    # yolo_dataset/
    #   images/Train, images/Val, images/Unlabeled
    #   labels/Train, labels/Val
    train_images_dst = new_root / "images" / "Train"
    val_images_dst = new_root / "images" / "Val"
    unlabeled_images_dst = new_root / "images" / "Unlabeled"

    train_labels_dst = new_root / "labels" / "Train"
    val_labels_dst = new_root / "labels" / "Val"

    logging.info(f"Создаём YOLO-датасет в: {new_root}")
    for d in [train_images_dst, val_images_dst, unlabeled_images_dst,
              train_labels_dst, val_labels_dst]:
        d.mkdir(parents=True, exist_ok=True)

    # === TRAIN ===
    logging.info("=== Обработка TRAIN ===")
    coco_to_yolo_split(
        annotations_path=train_ann_path,
        src_images_dir=train_images_src,
        dst_images_dir=train_images_dst,
        dst_labels_dir=train_labels_dst,
        class_name=class_name,
        yolo_class_id=0,
        move_images=MOVE_IMAGES,
    )

    # === VAL ===
    logging.info("=== Обработка VAL ===")
    coco_to_yolo_split(
        annotations_path=val_ann_path,
        src_images_dir=val_images_src,
        dst_images_dir=val_images_dst,
        dst_labels_dir=val_labels_dst,
        class_name=class_name,
        yolo_class_id=0,
        move_images=MOVE_IMAGES,
    )

    # === UNLABELED (если есть) ===
    if unlabeled_images_src is not None:
        logging.info("=== Обработка UNLABELED ===")
        copy_unlabeled_images(
            src_images_dir=unlabeled_images_src,
            dst_images_dir=unlabeled_images_dst,
            move_images=MOVE_IMAGES,
        )
    else:
        logging.info("unlabeled в конфиге не задан — пропускаем этот шаг.")

    logging.info("Готово! YOLO-датасет собран.")
    logging.info(
        "Структура:\n"
        f"{new_root}/images/Train\n"
        f"{new_root}/images/Val\n"
        f"{new_root}/images/Unlabeled\n"
        f"{new_root}/labels/Train\n"
        f"{new_root}/labels/Val"
    )


if __name__ == "__main__":
    build_yolo_dataset()
