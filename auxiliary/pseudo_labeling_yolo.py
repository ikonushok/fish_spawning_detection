# src/pseudo_labeling_yolo.py
# псевдо-разметка YOLO
from pathlib import Path
from typing import List, Tuple
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm, trange


def list_unlabeled_images(unlabeled_dir: str | Path) -> list[Path]:
    unlabeled_dir = Path(unlabeled_dir)
    paths = []
    for p in unlabeled_dir.glob("*.*"):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            paths.append(p)
    return paths


def compute_image_scores_yolo(
        model: YOLO,
        unlabeled_images_dir: str | Path,
        score_mode: str = "min",
        batch_size: int = 16  # Добавляем параметр для размера батча
) -> List[Tuple[Path, float, object]]:
    """
    Возвращает список (img_path, frame_score, result)
    img_path — путь к ИСХОДНОМУ unlabeled изображению.
    """
    img_paths = list_unlabeled_images(unlabeled_images_dir)
    scores = []

    # Разбиваем изображения на батчи
    for i in trange(0, len(img_paths), batch_size, desc="Pseudo-labeling.."):
        batch_paths = img_paths[i:i + batch_size]

        # Прогоняем изображения по батчу
        results = model.predict(
            source=[str(p) for p in batch_paths],
            stream=True,
            conf=0.0,
            verbose=False,
        )

        # Обрабатываем результаты для текущего батча
        for img_path, r in zip(batch_paths, results):
        # for img_path, r in tqdm(
        #         zip(batch_paths, results),
        #         total=len(batch_paths),
        #         desc=f"Pseudo-labeling (batch {i // batch_size + 1})",
        # ):
            if r.boxes is None or r.boxes.shape[0] == 0:
                frame_score = 0.0
            else:
                confs = r.boxes.conf.cpu().numpy()
                if score_mode == "min":
                    frame_score = float(confs.min())
                elif score_mode == "mean":
                    frame_score = float(confs.mean())
                else:
                    frame_score = float(confs.max())

            # Добавляем результат
            scores.append((img_path, frame_score, r))  # Сохраняем r

    return scores


def select_threshold_by_percentile(
    scores: List[Tuple[Path, float, object]],
    percentile: float,
) -> float:
    vals = np.array([s[1] for s in scores])
    return float(np.percentile(vals, percentile))


def save_pseudo_labels_yolo(
    scores: List[Tuple[Path, float, object]],
    thr: float,
    max_images: int,
    pseudo_labels_dir: str | Path,
    pseudo_images_dir: str | Path,
) -> list[Path]:
    """
    Сохраняет псевдо-разметку:
      - копирует картинки в pseudo_images_dir
      - пишет txt в pseudo_labels_dir
    Возвращает список ИСХОДНЫХ путей к изображениями из unlabeled,
    чтобы потом их можно было удалить.
    """
    from shutil import copy2

    pseudo_labels_dir = Path(pseudo_labels_dir)
    pseudo_images_dir = Path(pseudo_images_dir)
    pseudo_labels_dir.mkdir(parents=True, exist_ok=True)
    pseudo_images_dir.mkdir(parents=True, exist_ok=True)

    selected = [(p, sc, r) for (p, sc, r) in scores if sc >= thr]
    selected = sorted(selected, key=lambda x: x[1], reverse=True)[:max_images]

    used_original_images: list[Path] = []

    for img_path, sc, r in selected:
        # копируем картинку в папку pseudo_images_dir
        out_img_path = pseudo_images_dir / img_path.name
        if not out_img_path.exists():
            copy2(img_path, out_img_path)

        boxes_xywhn = r.boxes.xywhn.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        label_path = pseudo_labels_dir / (img_path.stem + ".txt")
        with label_path.open("w") as f:
            for cls_id, (xc, yc, w, h) in zip(cls_ids, boxes_xywhn):
                f.write(f"{int(cls_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        used_original_images.append(img_path)

    return used_original_images

