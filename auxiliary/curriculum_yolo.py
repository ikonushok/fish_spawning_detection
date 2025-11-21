# auxiliary/curriculum_yolo.py
import os
import yaml
import shutil
import logging

from pathlib import Path
from shutil import copy2
from ultralytics import YOLO
from PIL import Image

from auxiliary.config_loader import load_config
from auxiliary.pseudo_labeling_yolo import (
    compute_image_scores_yolo,
    select_threshold_by_percentile,
    save_pseudo_labels_yolo,
    list_unlabeled_images,
)

# === КОРЕНЬ ПРОЕКТА И ПУТЬ ДО КОНФИГА ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (PROJECT_ROOT / "configs" / "config.yaml").resolve()

# Базовая настройка логгера (если ещё не настроен где-то выше)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
    dataset_root: Path,
) -> Path:
    """
    Собираем data_iter_k.yaml для Ultralytics:
      path: корень YOLO-датасета (dataset/yolo_dataset)
      train: абсолютный путь к train_images_dir (iter_k/train_images)
      val:   абсолютный путь к val_images_dir (обычно images/Val внутри root_dir)
    """
    data_cfg = {
        "path": str(dataset_root),  # /.../dataset/yolo_dataset
        "train": str(train_images_dir),
        "val": str(val_images_dir),
        "names": {i: name for i, name in enumerate(cfg["classes"])},
    }
    out_yaml = out_dir / f"data_iter_{iter_idx}.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False, allow_unicode=True)
    return out_yaml


def verify_labels(train_images_dir: Path, train_labels_dir: Path) -> int:
    """
    Проверяет все изображения и метки в директориях train_images_dir и train_labels_dir:
    1. Для каждого изображения ищет соответствующую метку.
    2. Метка не должна быть пустой.
    3. Формат каждой строки: class_id x_center y_center width height
       - координаты в пределах [0, 1]
       - width, height > 0
    Возвращает общее количество ошибок.
    """
    image_paths = list(train_images_dir.glob("*.*"))

    total_images = 0
    errors = 0

    missing_label = 0
    empty_label = 0
    bad_format = 0

    # Чтобы не заспамить лог, запомним по несколько примеров каждого типа
    examples_missing = []
    examples_empty = []
    examples_bad = []

    for img_path in image_paths:
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        total_images += 1
        label_path = train_labels_dir / (img_path.stem + ".txt")

        # 1. Метка отсутствует
        if not label_path.exists():
            errors += 1
            missing_label += 1
            if len(examples_missing) < 10:
                examples_missing.append(img_path.name)
            continue

        # 2. Метка пустая
        with open(label_path, "r") as f:
            labels = [ln.strip() for ln in f.readlines() if ln.strip()]

        if not labels:
            errors += 1
            empty_label += 1
            if len(examples_empty) < 10:
                examples_empty.append(img_path.name)
            continue

        # 3. Проверка формата
        for label in labels:
            parts = label.split()
            if len(parts) < 5:
                errors += 1
                bad_format += 1
                if len(examples_bad) < 10:
                    examples_bad.append(f"{img_path.name}: '{label}'")
                break
            try:
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and width > 0 and height > 0):
                    errors += 1
                    bad_format += 1
                    if len(examples_bad) < 10:
                        examples_bad.append(f"{img_path.name}: '{label}'")
                    break

            except ValueError:
                errors += 1
                bad_format += 1
                if len(examples_bad) < 10:
                    examples_bad.append(f"{img_path.name}: '{label}'")
                break

    logging.debug(
        f"[VERIFY] Всего изображений: {total_images}, ошибок: {errors} "
        f"(нет метки: {missing_label}, пустая: {empty_label}, неверный формат: {bad_format})"
    )

    if examples_missing:
        logging.warning("[VERIFY] Примеры картинок без меток: " + ", ".join(examples_missing))
    if examples_empty:
        logging.warning("[VERIFY] Примеры картинок с пустыми метками: " + ", ".join(examples_empty))
    if examples_bad:
        logging.warning("[VERIFY] Примеры картинок с некорректными метками:\n" + "\n".join(examples_bad))

    return errors


def copy_image_if_needed(src: Path, dest: Path):
    """Проверка, существует ли файл, прежде чем копировать."""
    if not dest.exists():
        copy2(src, dest)
        logging.debug(f"Изображение {src.name} скопировано в {dest}")
    else:
        logging.debug(f"Изображение {src.name} уже существует в {dest}")


def remove_previous_iteration_data(iter_idx: int, work_dir: Path):
    """Удаляет данные предыдущей итерации, чтобы освободить место на диске."""
    prev_iter_dir = work_dir / f"iter_{iter_idx - 1}"
    if prev_iter_dir.exists():
        print(f"Удаление старых данных итерации {iter_idx - 1}...")
        shutil.rmtree(f'{prev_iter_dir}/images')  # Удаляет всю директорию с данными итерации
        shutil.rmtree(f'{prev_iter_dir}/labels')  # Удаляет всю директорию с данными итерации
        shutil.rmtree(f'{prev_iter_dir}/weights')  # Удаляет всю директорию с данными итерации
        # print(f"Данные итерации {iter_idx - 1} удалены.")


def move_image_to_used(src: Path, dest: Path):
    """Перемещает изображение из папки Unlabeled в папку Unlabeled_used."""
    try:
        # Если целевая папка не существует, создаем её
        dest.mkdir(parents=True, exist_ok=True)
        # Перемещаем файл в новую папку
        shutil.move(str(src), str(dest / src.name))  # Перемещаем файл
        logging.debug(f"Изображение {src.name} перемещено в {dest}")
    except Exception as e:
        logging.error(f"Ошибка при перемещении {src.name} в {dest}: {e}")


def merge_train_dataset_for_iter(cfg: dict, root_dir: Path, work_dir: Path, iter_idx: int):
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

    # Стандартная структура для YOLO
    train_images_dir = iter_dir / "images"
    train_labels_dir = iter_dir / "labels"

    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    # 1) Исходные размеченные (из нового YOLO-датасета)
    # root_dir = dataset/yolo_dataset
    original_images_dir = root_dir / cfg["dataset"]["train"]["images_dir"]  # images/Train
    original_labels_root = root_dir / cfg["dataset"]["train"]["images_dir"].replace("images", "labels")
    logging.info(f"Ищу изображения в папке: {original_images_dir}")
    logging.info(f"Ищу метки в папке: {original_labels_root}")

    img_paths = list(original_images_dir.glob("*.*"))
    if img_paths:
        img_w, img_h = get_image_size(img_paths[0])
        logging.info(f"Размер изображений (пример): {img_w}x{img_h}")

    # Копируем только те изображения, для которых есть корректные метки
    for img_path in img_paths:
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label_path = original_labels_root / (img_path.stem + ".txt")

        if label_path.exists():
            with open(label_path, "r") as f:
                labels = f.readlines()

            if not labels:
                logging.warning(f"Метка для {img_path.name} пуста! Изображение не добавляется в датасет.")
                continue

            # Проверка формата меток
            valid_format = True
            # Для каждой метки:
            for label in labels:
                parts = label.split()
                if len(parts) < 5:
                    logging.error(f"Некорректная метка (недостаточно параметров): {label}")
                    continue  # Пропускаем метку, если не хватает параметров
                try:
                    # Преобразуем значения в числа
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Проверяем, что все значения находятся в допустимом диапазоне
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and width > 0 and height > 0):
                        logging.error(f"Некорректные значения в метке: {label}")
                        continue  # Пропускаем метку с некорректными значениями

                except ValueError:
                    logging.error(f"Ошибка при преобразовании значений метки для изображения {img_path.name}: {label}")
                    continue  # Если не удалось преобразовать значения, пропускаем метку

            if not valid_format:
                logging.warning(f"Некорректный формат метки для изображения {img_path.name}. Пропускаем.")
                continue

            out_img_path = train_images_dir / img_path.name
            out_label_path = train_labels_dir / label_path.name

            copy_image_if_needed(img_path, out_img_path)

            if not out_label_path.exists():
                copy2(label_path, out_label_path)
            logging.debug(f"Метка для {img_path.name} найдена и скопирована в {out_label_path}")
        else:
            logging.debug(f"Метка для {img_path.name} не найдена! Изображение не добавляется в датасет.")

    # Удаляем старый кеш (если имеется) и пересоздаём его
    cache_file = train_images_dir.with_suffix(".cache")  # .../iter_0/images.cache
    if cache_file.exists():
        os.remove(cache_file)
        logging.info(f"Старый кеш удалён: {cache_file}")

    # 2) Псевдоразметка из прошлых итераций
    for prev_it in range(iter_idx):
        prev_dir = work_dir / f"iter_{prev_it}"
        pseudo_images_dir = prev_dir / "images"
        pseudo_labels_dir = prev_dir / "labels"

        logging.info(f"Ищу псевдоразметку в папке: {pseudo_images_dir}")
        logging.info(f"Ищу метки псевдоразметки в папке: {pseudo_labels_dir}")

        if not pseudo_images_dir.exists():
            continue

        for img_path in pseudo_images_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            label_path = pseudo_labels_dir / (img_path.stem + ".txt")

            if label_path.exists():
                with open(label_path, "r") as f:
                    labels = f.readlines()

                if not labels:
                    logging.warning(f"Метка для {img_path.name} из псевдоразметки пуста! Изображение не добавляется.")
                    continue

                valid_format = True
                for label in labels:
                    parts = label.split()
                    if len(parts) < 5:
                        valid_format = False
                        break
                    try:
                        x_center, y_center, width, height = map(float, parts[1:5])
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and width > 0 and height > 0):
                            valid_format = False
                            break
                    except ValueError:
                        valid_format = False
                        break

                if not valid_format:
                    logging.warning(
                        f"Некорректный формат метки для изображения {img_path.name} из псевдоразметки. Пропускаем."
                    )
                    continue

                out_img_path = train_images_dir / img_path.name
                out_label_path = train_labels_dir / label_path.name

                copy_image_if_needed(img_path, out_img_path)

                if not out_label_path.exists():
                    copy2(label_path, out_label_path)
                logging.debug(f"Метка для {img_path.name} из псевдоразметки найдена и скопирована в {out_label_path}")
            else:
                logging.warning(f"Метка для {img_path.name} из псевдоразметки не найдена! Изображение не добавляется.")

    # Проверим, что метки для всех изображений существуют и корректны
    errors = verify_labels(train_images_dir, train_labels_dir)

    if errors > 0:
        logging.error(f"Обнаружены ошибки в датасете. Общее количество ошибок: {errors}")
        raise ValueError("Датасет не в порядке. Прекращаем выполнение.")

    logging.info(f"[ITER {iter_idx}] Датасет сформирован без ошибок.")
    logging.info(
        f"[ITER {iter_idx}] Сформирован train-датасет:\n"
        f"  train_images: {train_images_dir}\n  train_labels: {train_labels_dir}"
    )

    return train_images_dir, train_labels_dir

# После завершения тренировки на текущей итерации
# remove_previous_iteration_data(it, work_dir)


def get_train_val_stats(images_dir: Path, labels_dir: Path) -> dict:
    """
    Функция для вычисления статистики для train или val данных.
    Возвращает словарь с количеством картинок и меток.
    """
    image_paths = list(images_dir.glob("*.*"))
    label_paths = {label_path.stem for label_path in labels_dir.glob("*.txt")}

    total_images = len(image_paths)
    total_labels = len(label_paths)

    # Картинки без меток
    images_with_labels = sum(1 for img_path in image_paths if img_path.stem in label_paths)
    images_without_labels = total_images - images_with_labels

    # Метки без картинок
    labels_without_images = total_labels - images_with_labels

    stats = {
        "total_images": total_images,
        "total_labels": total_labels,
        "images_with_labels": images_with_labels,
        "images_without_labels": images_without_labels,
        "labels_without_images": labels_without_images
    }

    return stats


def curriculum_training(config_path: str | None = None):
    """
    Основной цикл Curriculum Learning поверх нового YOLO-датасета.
    """
    # 1) Определяем путь к конфигу
    if config_path is None:
        cfg_path = CONFIG_PATH
    else:
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = (PROJECT_ROOT / cfg_path).resolve()

    cfg = load_config(str(cfg_path))

    # 2) Путь к data_supervised.yaml
    data_supervised_rel = cfg["output"]["yolo_data_supervised"]
    data_supervised_path = (PROJECT_ROOT / data_supervised_rel).resolve()

    if not data_supervised_path.exists():
        raise FileNotFoundError(f"Не найден {data_supervised_path}. Убедитесь, что prepare_supervised.py создал его.")

    with data_supervised_path.open("r", encoding="utf-8") as f:
        data_supervised = yaml.safe_load(f)

    # 3) Корень YOLO-датасета
    root_dir_cfg = Path(cfg["dataset"]["root_dir"])
    if root_dir_cfg.is_absolute():
        root_dir = root_dir_cfg
    else:
        root_dir = (PROJECT_ROOT / root_dir_cfg).resolve()

    # Папки unlabeled и val
    unlabeled_dir = root_dir / cfg["dataset"]["unlabeled"]["images_dir"]
    val_images_dir = root_dir / cfg["dataset"]["val"]["images_dir"]
    val_labels_dir = root_dir / cfg["dataset"]["val"]["labels_dir"]  # Добавлен путь к меткам val

    # 4) Рабочая директория для Curriculum
    work_rel = cfg["output"]["curriculum_dir"]
    work_dir = (PROJECT_ROOT / work_rel).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    current_percentile = cfg["curriculum"]["start_percentile"]
    max_images_per_iter = cfg["curriculum"]["max_images_per_iter"]
    max_images_increase_factor = cfg["curriculum"]["max_images_increase_factor"]
    unlabeled_used_dir = root_dir / cfg["dataset"]["unlabeled"]["unlabeled_used"]

    for it in range(cfg["curriculum"]["num_iters"]):
        print(f"\n=== Curriculum iteration {it + 1}/{cfg['curriculum']['num_iters']} "
              f"(percentile={current_percentile}, max_images={max_images_per_iter}) ===")

        # Проверяем, остались ли неразмеченные кадры
        if not list_unlabeled_images(unlabeled_dir):
            print("Неразмеченных изображений больше нет. Стоп.")
            break

        iter_dir = work_dir / f"iter_{it}"
        pseudo_labels_dir = iter_dir / "labels"
        pseudo_images_dir = iter_dir / "images"

        # 1) Объединяем train (размеченные + все старые pseudo) в один датасет
        train_images_dir, train_labels_dir = merge_train_dataset_for_iter(  # Сохраняем train_labels_dir
            cfg=cfg,
            root_dir=root_dir,
            work_dir=work_dir,
            iter_idx=it,
        )

        # Получаем статистику для train
        train_stats = get_train_val_stats(train_images_dir, train_labels_dir)

        logging.info(f"===TRAIN STATS: ==="
                     f"\tИтерация {it + 1}:\tкартинок всего:{train_stats['total_images']}, "
                     f"меток всего: {train_stats['total_labels']}...\t"
                     f"картинок с метками: {train_stats['images_with_labels']}, "
                     f"картинок без меток: {train_stats['images_without_labels']}, "
                     f"меток без картинок: {train_stats['labels_without_images']}")

        # Получаем статистику для val
        val_stats = get_train_val_stats(val_images_dir, val_labels_dir)

        logging.info(f"=== VAL STATS ===\n"
                     f"\tИтерация {it + 1}:\tкартинок всего:{val_stats['total_images']}, "
                     f"меток всего: {val_stats['total_labels']}...\t"
                     f"картинок с метками: {val_stats['images_with_labels']}, "
                     f"картинок без меток: {val_stats['images_without_labels']}, "
                     f"меток без картинок: {val_stats['labels_without_images']}")

        # Логируем текущее количество изображений в train
        logging.debug(f"Итерация {it + 1}: Количество изображений в train "
                      f"до добавления псевдоразметки: {len(list(train_images_dir.glob('*')))}")

        # 2) Инициализируем модель из базового чекпоинта
        base_model = cfg["yolo"]["base_model"]
        model = YOLO(base_model)

        # 3) Генерим data_iter_k.yaml
        data_yaml_path = build_data_yaml_for_iter(
            cfg=cfg,
            iter_idx=it,
            train_images_dir=train_images_dir,
            val_images_dir=val_images_dir,
            out_dir=work_dir,
            dataset_root=root_dir,
        )

        # 4) Обучаем YOLO на текущем train
        model.train(
            data=str(data_yaml_path),
            imgsz=cfg["yolo"]["img_size"],
            epochs=cfg["yolo"]["epochs_per_iter"],
            batch=cfg["yolo"]["batch_size"],
            project=str(work_dir),
            name=f"iter_{it}",
            exist_ok=True,
            # Добавление параметров для аугментации данных
            augment=True,  # Включаем аугментацию
            flipud=0.5,  # Вертикальное отражение
            fliplr=0.5,  # Горизонтальное отражение
            scale=0.2,  # Масштабирование
            degrees=10,  # Повороты
        )

        # 5) Берём best.pt
        best_ckpt = list((work_dir / f"iter_{it}" / "weights").glob("best*.pt"))
        if best_ckpt:
            model = YOLO(str(best_ckpt[0]))

        # 6) Считаем scores для оставшихся unlabeled
        scores = compute_image_scores_yolo(
            model=model,
            unlabeled_images_dir=unlabeled_dir,
            score_mode=cfg["curriculum"]["score_mode"],
            batch_size=cfg["yolo"]["batch_size"]
        )

        thr = select_threshold_by_percentile(scores, current_percentile)
        print(f"Порог по перцентилю: {thr:.4f}")

        # 7) Сохраняем псевдо-разметку и получаем список использованных unlabeled-кадров
        used_original_images = save_pseudo_labels_yolo(
            scores=scores,
            thr=thr,
            max_images=max_images_per_iter,
            pseudo_labels_dir=pseudo_labels_dir,
            pseudo_images_dir=pseudo_images_dir,
        )
        # Логируем количество добавленных изображений
        logging.info(f"Итерация {it + 1}: Добавлено {len(used_original_images)} псевдоразмеченных изображений")

        # 8) Перемещаем использованные изображения в папку Unlabeled_used
        removed = 0
        for p in used_original_images:
            move_image_to_used(p, unlabeled_used_dir)
            removed += 1

        print(f"Перемещено использованных изображений: {removed}")
        logging.info(f"Итерация {it + 1}: Перемещено {removed} изображений в папку Unlabeled_used")

        # 9) Обновляем перцентиль
        current_percentile = max(0.0, current_percentile - cfg["curriculum"]["step_percentile"])

        # Увеличиваем max_images_per_iter на основе конфигурации
        if it < cfg["curriculum"]["num_iters"] - 1:  # Увеличиваем не в последней итерации
            max_images_per_iter = int(max_images_per_iter * max_images_increase_factor)
            max_images_per_iter = min(max_images_per_iter, len(list(unlabeled_dir.glob("*"))))




