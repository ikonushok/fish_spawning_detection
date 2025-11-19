import random
from pathlib import Path
import shutil

# Корень датасета
DATASET_ROOT = Path("/Users/bobrsubr/PycharmProjects/fish_spawning_detection/dataset/yolo_dataset")

IMAGES_TRAIN = DATASET_ROOT / "images" / "Train"
IMAGES_VAL = DATASET_ROOT / "images" / "Val"
LABELS_TRAIN = DATASET_ROOT / "labels" / "Train"
LABELS_VAL = DATASET_ROOT / "labels" / "Val"

# 👉 Сколько файлов переносим из Train в Val
N_TO_MOVE = 1000  # поменяй на нужное число

RANDOM_SEED = 42  # для воспроизводимости результата


def main():
    # Все картинки из Train (jpg/jpeg/png)
    train_images = [
        img for img in IMAGES_TRAIN.iterdir()
        if img.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    if not train_images:
        print("В Train нет изображений.")
        return

    # Сколько реально можем перенести (если N_TO_MOVE больше, чем есть файлов)
    n_move = min(N_TO_MOVE, len(train_images))

    random.seed(RANDOM_SEED)
    files_to_move = random.sample(train_images, n_move)

    print(f"Будет перенесено {n_move} файлов из Train -> Val")

    for img_path in files_to_move:
        label_path = LABELS_TRAIN / f"{img_path.stem}.txt"

        # пути назначения
        img_dest = IMAGES_VAL / img_path.name
        label_dest = LABELS_VAL / label_path.name

        # переносим изображение
        shutil.move(str(img_path), str(img_dest))
        print(f"Image moved: {img_path.name}")

        # если рядом есть метка — тоже переносим
        if label_path.exists():
            shutil.move(str(label_path), str(label_dest))
            print(f"Label moved: {label_path.name}")
        else:
            print(f"No label file for: {img_path.name}")

    print("Готово!")


if __name__ == "__main__":
    main()
