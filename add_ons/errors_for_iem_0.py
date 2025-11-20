from pathlib import Path

from auxiliary.config_loader import load_config
from auxiliary.curriculum_yolo import verify_labels

if __name__ == "__main__":
    # Загружаем конфиг
    cfg = load_config('../configs/config.yaml')
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Корень YOLO-датасета
    root_dir_cfg = Path(cfg["dataset"]["root_dir"])
    if root_dir_cfg.is_absolute():
        root_dir = root_dir_cfg
    else:
        root_dir = (PROJECT_ROOT / root_dir_cfg).resolve()

    # Рабочая директория curriculum
    work_rel = cfg["output"]["curriculum_dir"]
    work_dir = (PROJECT_ROOT / work_rel).resolve()

    # Какая итерация нас интересует
    iter_idx = 0
    iter_dir = work_dir / f"iter_{iter_idx}"
    train_images_dir = iter_dir / "train_images"
    train_labels_dir = iter_dir / "train_labels"

    print(f"Проверяю iter_{iter_idx}")
    print(f"  train_images_dir = {train_images_dir}")
    print(f"  train_labels_dir = {train_labels_dir}")

    errors = verify_labels(train_images_dir, train_labels_dir)
    print(f"Найдено ошибок: {errors}")
