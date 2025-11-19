# auxiliary/prepare_supervised.py
# базовое обучение по уже ГОТОВОМУ YOLO-датасету (без COCO-аннотаций)

import sys
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO

from .config_loader import load_config


# === ОПРЕДЕЛЯЕМ КОРЕНЬ ПРОЕКТА ===
# /Users/bobrsubr/PycharmProjects/fish_spawning_detection
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Абсолютный путь к configs/config.yaml
CONFIG_PATH = (PROJECT_ROOT / "configs" / "config.yaml").resolve()


# === НАСТРОЙКА ЛОГГЕРА ===
cfg_for_logging = load_config(str(CONFIG_PATH))

# в конфиге: "artefacts/training_log.txt"
log_file_rel = cfg_for_logging["output"]["log_file"]
log_file = (PROJECT_ROOT / log_file_rel).resolve()

log_path = log_file
log_path.parent.mkdir(parents=True, exist_ok=True)  # Создаём папки для лога, если их нет

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(file_handler)

# Если нужен вывод и в консоль — раскомментируй
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
# logging.getLogger().addHandler(console_handler)


def prepare_yolo_labels(cfg: dict, root_dir: Path) -> dict:
    """
    НОВАЯ версия:
    НИЧЕГО не конвертируем, просто:
      - находим папки с картинками (train/val) из конфига
      - находим соответствующие папки с лейблами (labels/Train, labels/Val)
    Структура:
        root_dir = dataset/yolo_dataset
        images/Train
        images/Val
        images/Unlabeled
        labels/Train
        labels/Val
    """
    train_cfg = cfg["dataset"]["train"]
    val_cfg = cfg["dataset"]["val"]

    # Папки с изображениями (абсолютные пути)
    train_images_dir = (root_dir / train_cfg["images_dir"]).resolve()
    val_images_dir = (root_dir / val_cfg["images_dir"]).resolve()

    # Имя сплита (Train / Val)
    train_split_name = Path(train_cfg["images_dir"]).name  # "Train"
    val_split_name = Path(val_cfg["images_dir"]).name      # "Val"

    # Папки с лейблами: root_dir/labels/<split_name>
    train_labels_dir = (root_dir / "labels" / train_split_name).resolve()
    val_labels_dir = (root_dir / "labels" / val_split_name).resolve()

    if not train_images_dir.exists():
        raise FileNotFoundError(f"TRAIN images dir not found: {train_images_dir}")
    if not val_images_dir.exists():
        raise FileNotFoundError(f"VAL images dir not found: {val_images_dir}")
    if not train_labels_dir.exists():
        raise FileNotFoundError(f"TRAIN labels dir not found: {train_labels_dir}")
    if not val_labels_dir.exists():
        raise FileNotFoundError(f"VAL labels dir not found: {val_labels_dir}")

    logging.info("\nИспользуем ГОТОВЫЙ YOLO-датасет, конвертацию COCO -> YOLO пропускаем.")
    logging.info(f"TRAIN images: {train_images_dir}")
    logging.info(f"TRAIN labels: {train_labels_dir}")
    logging.info(f"VAL   images: {val_images_dir}")
    logging.info(f"VAL   labels: {val_labels_dir}\n")

    return {
        "train_images": str(train_images_dir),
        "val_images": str(val_images_dir),
        "train_labels": str(train_labels_dir),
        "val_labels": str(val_labels_dir),
    }


def create_yolo_data_yaml(cfg: dict, paths: dict, root_dir: Path) -> Path:
    """
    Создаём data_supervised.yaml (путь берём из cfg["output"]["yolo_data_supervised"]).

    В конфиге:
      yolo_data_supervised: "artefacts/data_supervised.yaml"

    Реальный путь:
      /.../fish_spawning_detection/artefacts/data_supervised.yaml
    """
    data_yaml_rel = cfg["output"]["yolo_data_supervised"]
    data_yaml_path = (PROJECT_ROOT / data_yaml_rel).resolve()

    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    names = cfg["classes"]

    train_rel = cfg["dataset"]["train"]["images_dir"]  # "images/Train"
    val_rel = cfg["dataset"]["val"]["images_dir"]      # "images/Val"

    yolo_data = {
        "path": str(root_dir),         # /.../dataset/yolo_dataset
        "train": train_rel,            # "images/Train"
        "val": val_rel,                # "images/Val"
        "names": {i: name for i, name in enumerate(names)},
    }

    with data_yaml_path.open("w") as f:
        yaml.safe_dump(yolo_data, f, sort_keys=False, allow_unicode=True)

    logging.info(f"Создан файл data_supervised.yaml: {data_yaml_path}")
    logging.info(f"Содержимое data.yaml: {yolo_data}")

    return data_yaml_path


def log_split_stats(split_name: str, images_dir: Path, labels_dir: Path) -> None:
    """
    Логирует статистику по сплиту:
      - всего картинок
      - всего txt-меток
      - картинок с метками
      - картинок без меток
    """
    img_paths = [p for p in images_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    label_paths = [p for p in labels_dir.iterdir()
                   if p.is_file() and p.suffix.lower() == ".txt"]

    img_set = {p.stem for p in img_paths}
    label_set = {p.stem for p in label_paths}

    images_with_labels = img_set & label_set
    images_without_labels = img_set - label_set
    labels_without_images = label_set - img_set

    logging.info(
        f"{split_name.upper()} STATS: "
        f"картинок всего: {len(img_paths)}, "
        f"меток всего: {len(label_paths)}, "
        f"картинок с метками: {len(images_with_labels)}, "
        f"картинок без меток: {len(images_without_labels)}, "
        f"меток без картинок: {len(labels_without_images)}\n"
    )


def train_supervised(cfg_path: str | None = None):
    """
    Если cfg_path не задан, используем CONFIG_PATH (configs/config.yaml в корне проекта).
    """
    if cfg_path is None:
        cfg_path = str(CONFIG_PATH)
    else:
        # если передали относительный путь — считаем его от PROJECT_ROOT
        cfg_path = str((PROJECT_ROOT / cfg_path).resolve()) if not Path(cfg_path).is_absolute() else cfg_path

    logging.info(f"Запуск тренировки на конфиге: {cfg_path}")
    cfg = load_config(cfg_path)

    # dataset.root_dir: "dataset/yolo_dataset" из конфига
    root_dir_cfg = Path(cfg["dataset"]["root_dir"])
    if root_dir_cfg.is_absolute():
        root_dir = root_dir_cfg
    else:
        root_dir = (PROJECT_ROOT / root_dir_cfg).resolve()

    logging.info(f"Корень YOLO-датасета: {root_dir}")

    # 1) Проверяем папки и собираем пути
    paths = prepare_yolo_labels(cfg, root_dir)

    # 1.1) Логируем статистику по Train и Val
    train_images_dir = Path(paths["train_images"])
    train_labels_dir = Path(paths["train_labels"])
    val_images_dir = Path(paths["val_images"])
    val_labels_dir = Path(paths["val_labels"])

    log_split_stats("train", train_images_dir, train_labels_dir)
    log_split_stats("val", val_images_dir, val_labels_dir)

    # 2) Создаём data_supervised.yaml в PROJECT_ROOT/artefacts/...
    data_yaml_path = create_yolo_data_yaml(cfg, paths, root_dir)

    # 3) Обучение YOLO
    logging.info(f"Запуск обучения модели с data.yaml: {data_yaml_path}")
    model = YOLO(cfg["yolo"]["base_model"])

    # runs_supervised: "artefacts/runs/supervised_yolo" -> абсолютный путь
    runs_rel = cfg["output"]["runs_supervised"]
    runs_dir = (PROJECT_ROOT / runs_rel).resolve()
    runs_dir.parent.mkdir(parents=True, exist_ok=True)

    model.train(
        data=str(data_yaml_path),
        imgsz=cfg["yolo"]["img_size"],
        epochs=cfg["yolo"]["epochs_supervised"],
        batch=cfg["yolo"]["batch_size"],
        project=str(runs_dir),
        name="baseline",
    )

    logging.info("Базовое обучение завершено.")


if __name__ == "__main__":
    train_supervised()
