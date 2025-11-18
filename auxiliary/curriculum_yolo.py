from pathlib import Path
from shutil import copy2
from ultralytics import YOLO
import yaml
from PIL import Image  # Подключаем Pillow для работы с изображениями

from .config_loader import load_config
from .pseudo_labeling_yolo import (
    compute_image_scores_yolo,
    select_threshold_by_percentile,
    save_pseudo_labels_yolo,
    list_unlabeled_images,
)

def get_image_size(image_path: Path) -> tuple[int, int]:
    """Получаем размеры изображения (ширина, высота)."""
    with Image.open(image_path) as img:
        return img.size

def build_data_yaml_for_iter(
    cfg: dict,
    iter_idx: int,
    train_images_dir: Path,
    val_images_dir: Path,
    out_dir: Path,
) -> Path:
    data_cfg = {
        "path": str(Path(cfg["dataset"]["root_dir"]).resolve()),
        "train": str(train_images_dir),
        "val": str(val_images_dir),
        "names": {i: name for i, name in enumerate(cfg["classes"])},
    }
    out_yaml = out_dir / f"data_iter_{iter_idx}.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w") as f:
        yaml.safe_dump(data_cfg, f)
    return out_yaml


def copy_image_if_needed(src, dest):
    """Проверка, существует ли файл, прежде чем копировать"""
    if not dest.exists():
        copy2(src, dest)


def merge_train_dataset_for_iter(
    cfg: dict,
    root_dir: Path,
    work_dir: Path,
    iter_idx: int,
) -> tuple[Path, Path]:
    """
    Создаёт для итерации iter_idx:
      - iter_k/train_images
      - iter_k/train_labels
    и копирует туда:
      - исходные размеченные train изображения + их YOLO-метки
      - все pseudo изображения + метки из СТАРЫХ итераций (0..iter_idx-1).

    Возвращает (train_images_dir, train_labels_dir).
    """
    iter_dir = work_dir / f"iter_{iter_idx}"
    train_images_dir = iter_dir / "train_images"
    train_labels_dir = iter_dir / "train_labels"

    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    # 1) исходные размеченные
    original_images_dir = root_dir / cfg["dataset"]["train"]["images_dir"]
    original_labels_root = root_dir / cfg["output"]["labels_dir"] / "train"

    # Извлекаем размеры изображений один раз
    img_paths = list(original_images_dir.glob("*.*"))
    if img_paths:
        img_w, img_h = get_image_size(img_paths[0])  # Используем первое изображение для получения размера

    for img_path in img_paths:
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        out_img_path = train_images_dir / img_path.name
        copy_image_if_needed(img_path, out_img_path)

        label_path = original_labels_root / (img_path.stem + ".txt")
        if label_path.exists():
            out_label_path = train_labels_dir / label_path.name
            if not out_label_path.exists():
                copy2(label_path, out_label_path)

    # 2) pseudo из прошлых итераций
    for prev_it in range(iter_idx):
        prev_dir = work_dir / f"iter_{prev_it}"
        pseudo_images_dir = prev_dir / "images"
        pseudo_labels_dir = prev_dir / "labels"

        if not pseudo_images_dir.exists():
            continue

        for img_path in pseudo_images_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            out_img_path = train_images_dir / img_path.name
            copy_image_if_needed(img_path, out_img_path)

            label_path = pseudo_labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                out_label_path = train_labels_dir / label_path.name
                if not out_label_path.exists():
                    copy2(label_path, out_label_path)

    print(
        f"[ITER {iter_idx}] Сформирован train-датасет:\n"
        f"  train_images: {train_images_dir}\n"
        f"  train_labels: {train_labels_dir}"
    )

    return train_images_dir, train_labels_dir


def curriculum_training(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    root_dir = Path(cfg["dataset"]["root_dir"]).resolve()

    unlabeled_dir = Path(cfg["dataset"]["unlabeled"]["images_dir"])
    val_images_dir = root_dir / cfg["dataset"]["val"]["images_dir"]

    work_dir = Path(cfg["output"]["curriculum_dir"]).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    current_percentile = cfg["curriculum"]["start_percentile"]

    for it in range(cfg["curriculum"]["num_iters"]):
        print(
            f"\n=== Curriculum iteration {it+1}/{cfg['curriculum']['num_iters']} "
            f"(percentile={current_percentile}) ==="
        )

        # проверяем, остались ли неразмеченные кадры
        if not list_unlabeled_images(unlabeled_dir):
            print("Неразмеченных изображений больше нет. Стоп.")
            break

        iter_dir = work_dir / f"iter_{it}"
        pseudo_labels_dir = iter_dir / "labels"
        pseudo_images_dir = iter_dir / "images"

        # 1) объединяем train (размеченные + все старые pseudo) в один датасет
        train_images_dir, _ = merge_train_dataset_for_iter(
            cfg=cfg,
            root_dir=root_dir,
            work_dir=work_dir,
            iter_idx=it,
        )

        # 2) инициализируем модель из базового чекпоинта (как в статье)
        base_model = cfg["yolo"]["base_model"]
        model = YOLO(base_model)

        # 3) генерим data_iter_k.yaml
        data_yaml_path = build_data_yaml_for_iter(
            cfg=cfg,
            iter_idx=it,
            train_images_dir=train_images_dir,
            val_images_dir=val_images_dir,
            out_dir=work_dir,
        )

        # 4) обучаем YOLO на текущем train
        model.train(
            data=str(data_yaml_path),
            imgsz=cfg["yolo"]["img_size"],
            epochs=cfg["yolo"]["epochs_per_iter"],
            batch=cfg["yolo"]["batch_size"],
            project=str(work_dir),
            name=f"iter_{it}",
            exist_ok=True,
        )

        # 5) берём best.pt
        best_ckpt = list((work_dir / f"iter_{it}" / "weights").glob("best*.pt"))
        if best_ckpt:
            model = YOLO(str(best_ckpt[0]))

        # 6) считаем scores для оставшихся unlabeled
        scores = compute_image_scores_yolo(
            model=model,
            unlabeled_images_dir=unlabeled_dir,
            score_mode=cfg["curriculum"]["score_mode"],
        )

        thr = select_threshold_by_percentile(scores, current_percentile)
        print(f"Порог по перцентилю: {thr:.4f}")

        # 7) сохраняем псевдо-разметку и получаем список исходных unlabeled-кадров
        used_original_images = save_pseudo_labels_yolo(
            scores=scores,
            thr=thr,
            max_images=cfg["curriculum"]["max_images_per_iter"],
            pseudo_labels_dir=pseudo_labels_dir,
            pseudo_images_dir=pseudo_images_dir,
        )
        print(f"Добавлено псевдо-размеченных кадров: {len(used_original_images)}")

        # 8) удаляем использованные unlabeled-картинки,
        #    чтобы не размечать их повторно на следующих итерациях
        removed = 0
        for p in used_original_images:
            try:
                p.unlink()
                removed += 1
            except FileNotFoundError:
                pass
        print(f"Удалено использованных unlabeled изображений: {removed}")

        # 9) обновляем перцентиль
        current_percentile = max(0.0, current_percentile - cfg["curriculum"]["step_percentile"])
