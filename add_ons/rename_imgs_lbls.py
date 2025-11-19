# add-ons/rename_frames.py
from pathlib import Path
from typing import List, Tuple

from auxiliary.config_loader import load_config

# Корень проекта: /Users/bobrsubr/PycharmProjects/fish_spawning_detection
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (PROJECT_ROOT / "configs" / "config.yaml").resolve()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(folder: Path) -> List[Path]:
    """Собираем список файлов-картинок в папке (без рекурсии), отсортированный по имени."""
    if not folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder}")
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    imgs.sort(key=lambda p: p.name)
    return imgs


def rename_split_with_labels(
    images_dir: Path,
    labels_dir: Path,
    start_idx: int,
) -> int:
    """
    Переименовать изображения и соответствующие метки:
      images_dir: папка с картинками (Train или Val)
      labels_dir: папка с txt-метками
      start_idx: с какого номера начинать (frame_<start_idx>)
    Возвращает следующий свободный индекс.
    """
    print(f"\n=== Переименование сплита ===")
    print(f"images_dir = {images_dir}")
    print(f"labels_dir = {labels_dir}")
    print(f"start_idx  = {start_idx}")

    images = collect_images(images_dir)

    # План переименования: [(old_img, new_img, old_label, new_label), ...]
    plan: List[Tuple[Path, Path, Path, Path]] = []
    idx = start_idx

    for img_path in images:
        old_stem = img_path.stem
        new_stem = f"frame_{idx:06d}"

        new_img = img_path.with_name(new_stem + img_path.suffix.lower())

        old_label = labels_dir / f"{old_stem}.txt"
        new_label = labels_dir / f"{new_stem}.txt"

        plan.append((img_path, new_img, old_label, new_label))
        idx += 1

    # Шаг 1 — переименовываем всё во временные имена, чтобы не было конфликтов
    # temp-имя: "__tmp__" + старое имя
    for old_img, _new_img, old_label, _new_label in plan:
        tmp_img = old_img.with_name("__tmp__" + old_img.name)
        if not tmp_img.exists():
            old_img.rename(tmp_img)

        if old_label.exists():
            tmp_label = old_label.with_name("__tmp__" + old_label.name)
            if not tmp_label.exists():
                old_label.rename(tmp_label)

    # Шаг 2 — из временных имён в конечные frame_XXXXXX
    for old_img, new_img, old_label, new_label in plan:
        tmp_img = old_img.withname("__tmp__" + old_img.name) if hasattr(old_img, "withname") else old_img.with_name("__tmp__" + old_img.name)
        # для совместимости с Python <3.12 — используем with_name
        tmp_img = old_img.with_name("__tmp__" + old_img.name)
        tmp_img.rename(new_img)

        tmp_label = old_label.with_name("__tmp__" + old_label.name)
        if tmp_label.exists():
            tmp_label.rename(new_label)

    print(f"Переименовано {len(images)} изображений. Следующий индекс: {idx}")
    return idx


def rename_unlabeled(images_dir: Path, start_idx: int) -> int:
    """
    Переименовать только картинки (без меток) в images/Unlabeled.
    """
    print(f"\n=== Переименование unlabeled ===")
    print(f"images_dir = {images_dir}")
    print(f"start_idx  = {start_idx}")

    images = collect_images(images_dir)

    plan: List[Tuple[Path, Path]] = []
    idx = start_idx

    for img_path in images:
        new_stem = f"frame_{idx:06d}"
        new_img = img_path.with_name(new_stem + img_path.suffix.lower())
        plan.append((img_path, new_img))
        idx += 1

    # temp → final
    for old_img, _new_img in plan:
        tmp_img = old_img.with_name("__tmp__" + old_img.name)
        if not tmp_img.exists():
            old_img.rename(tmp_img)

    for old_img, new_img in plan:
        tmp_img = old_img.with_name("__tmp__" + old_img.name)
        tmp_img.rename(new_img)

    print(f"Переименовано {len(images)} изображений в unlabeled. Финальный индекс: {idx}")
    return idx


def main():
    cfg = load_config(str(CONFIG_PATH))

    # Корень нового YOLO-датасета: dataset/yolo_dataset
    root_dir_cfg = Path(cfg["dataset"]["root_dir"])  # "dataset/yolo_dataset"
    if root_dir_cfg.is_absolute():
        root_dir = root_dir_cfg
    else:
        root_dir = (PROJECT_ROOT / root_dir_cfg).resolve()

    print(f"Корень YOLO-датасета: {root_dir}")

    # Папки сплитов
    train_images_dir = root_dir / cfg["dataset"]["train"]["images_dir"]      # images/Train
    val_images_dir = root_dir / cfg["dataset"]["val"]["images_dir"]          # images/Val
    unlabeled_images_dir = root_dir / cfg["dataset"]["unlabeled"]["images_dir"]  # images/Unlabeled

    train_labels_dir = root_dir / "labels" / "Train"
    val_labels_dir = root_dir / "labels" / "Val"

    # Проверка, что всё существует
    for p in [train_images_dir, val_images_dir, unlabeled_images_dir, train_labels_dir, val_labels_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Ожидаемая папка не найдена: {p}")

    idx = 0

    # 1) Train
    idx = rename_split_with_labels(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        start_idx=idx,
    )

    # 2) Val
    idx = rename_split_with_labels(
        images_dir=val_images_dir,
        labels_dir=val_labels_dir,
        start_idx=idx,
    )

    # 3) Unlabeled
    idx = rename_unlabeled(
        images_dir=unlabeled_images_dir,
        start_idx=idx,
    )

    print(f"\nГотово. Финальный последний индекс: {idx - 1}")


if __name__ == "__main__":
    main()
