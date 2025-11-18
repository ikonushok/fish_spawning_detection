# src/prepare_supervised.py
# конвертация + базовое обучение
import sys
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO

from .config_loader import load_config
from .coco_to_yolo import convert_coco_to_yolo_for_split


# Настройка логгера
cfg = load_config("configs/config.yaml")
log_file = cfg["output"]["log_file"]  # Берём путь из конфигурации
log_path = Path(log_file)
log_path.parent.mkdir(parents=True, exist_ok=True)  # Создаём папки для лога, если их нет

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Создаём обработчик для записи в файл
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(file_handler)

# Создаём обработчик для вывода в консоль
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
# logging.getLogger().addHandler(console_handler)



def prepare_yolo_labels(cfg: dict, root_dir: Path) -> dict:
    """
    Конвертируем COCO-аннотации для train/val в YOLO txt.
    Пишем ЛЕЙБЛЫ туда, где их ждёт YOLO:
      job_181.../labels/default
      job_191.../labels/default
    """
    # logging.debug("=== Конвертация TRAIN (instances_default.json) ===")
    class_name = cfg['classes'][0]
    train_cfg = cfg["dataset"]["train"]
    val_cfg = cfg["dataset"]["val"]

    # Папки с картинками
    train_images_dir = (root_dir / train_cfg["images_dir"]).resolve()
    val_images_dir = (root_dir / val_cfg["images_dir"]).resolve()

    # job_* уровень: .../job_181_dataset_..._coco 1
    train_job_dir = train_images_dir.parents[1]  # default -> images -> job_181...
    val_job_dir = val_images_dir.parents[1]  # default -> images -> job_191...

    # Куда YOLO будет смотреть за лейблами:
    #   job_181.../labels/default
    #   job_191.../labels/default
    train_labels_dir = train_job_dir / "labels" / train_images_dir.name
    val_labels_dir = val_job_dir / "labels" / val_images_dir.name

    # Создание папок, если их нет
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    logging.info("\n=== Конвертация TRAIN (instances_default.json) ===")
    train_stats = convert_coco_to_yolo_for_split(
        images_dir=train_images_dir,
        annotations_path=root_dir / train_cfg["annotations_file"],
        labels_out_dir=train_labels_dir,
        class_name=class_name,
        yolo_class_id=0,
    )
    logging.info(
        f"\nTRAIN: всего картинок:         {train_stats['total_images']}\n"
        f"TRAIN: картинок с рыбами:      {train_stats['images_with_fish']}\n"
        f"TRAIN: всего боксов рыб:       {train_stats['total_fish_boxes']}\n"
        f"TRAIN: отсутствующих картинок: {train_stats['missing_images']}\n"
    )

    logging.info("\n=== Конвертация VAL (instances_default.json) ===")
    val_stats = convert_coco_to_yolo_for_split(
        images_dir=val_images_dir,
        annotations_path=root_dir / val_cfg["annotations_file"],
        labels_out_dir=val_labels_dir,
        class_name=class_name,
        yolo_class_id=0,
    )
    logging.info(
        f"\nVAL:   всего картинок:         {val_stats['total_images']}\n"
        f"VAL:   картинок с рыбами:      {val_stats['images_with_fish']}\n"
        f"VAL:   всего боксов рыб:       {val_stats['total_fish_boxes']}\n"
        f"VAL:   отсутствующих картинок: {val_stats['missing_images']}\n"
    )

    return {
        "train_images": str(train_images_dir),
        "val_images": str(val_images_dir),
        "train_labels": str(train_labels_dir),
        "val_labels": str(val_labels_dir),
    }


def create_yolo_data_yaml(cfg: dict, paths: dict, root_dir: Path) -> Path:
    """
    Создаём data_supervised.yaml.
    Сначала проверяем, есть ли папка для сохранения файла.
    Если нет — создаём.
    """
    data_yaml_path = root_dir / cfg["output"]["yolo_data_supervised"]

    # Создание папки, если её нет
    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # Строим yaml
    names = cfg["classes"]
    yolo_data = {
        "path": str(root_dir),
        "train": paths["train_images"],
        "val": paths["val_images"],
        "names": {i: name for i, name in enumerate(names)},
    }

    with data_yaml_path.open("w") as f:
        yaml.safe_dump(yolo_data, f)

    logging.info(f"Создан файл: {data_yaml_path}")

    return data_yaml_path


def train_supervised(cfg_path: str = "config.yaml"):
    logging.debug(f"Запуск тренировки на конфиге: {cfg_path}")
    cfg = load_config(cfg_path)
    root_dir = Path(cfg["dataset"]["root_dir"]).resolve()

    # 1) COCO -> YOLO txt
    paths = prepare_yolo_labels(cfg, root_dir)

    # 2) data_supervised.yaml
    data_yaml_path = create_yolo_data_yaml(cfg, paths, root_dir)

    # 3) Обучение YOLO только на размеченных данных (шаг 0)
    logging.info(f"Запуск обучения модели с путём данных: {data_yaml_path}")
    model = YOLO(cfg["yolo"]["base_model"])
    model.train(
        data=str(data_yaml_path),
        imgsz=cfg["yolo"]["img_size"],
        epochs=cfg["yolo"]["epochs_supervised"],
        batch=cfg["yolo"]["batch_size"],
        project=cfg["output"]["runs_supervised"],
        name="baseline",
    )

    logging.info("Базовое обучение завершено.")


if __name__ == "__main__":
    train_supervised()
