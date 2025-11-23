import os
import cv2
import yaml
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# Функция для загрузки конфигурации
def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Функция для получения последней обученной модели
def get_latest_model(cfg: dict) -> str:
    # Путь к папке с результатами
    runs_dir = Path(cfg["output"]["runs_supervised"])
    if not runs_dir.exists():
        raise FileNotFoundError(f"Папка {runs_dir} не найдена.")

    # Находим самую последнюю модель
    model_dir = max(runs_dir.glob("*/weights/best.pt"), key=os.path.getmtime, default=None)
    if model_dir is None:
        raise FileNotFoundError("Последняя модель не найдена.")

    return str(model_dir)

# Функция для выполнения предсказания на изображениях
def predict_on_unlabeled_images(cfg: dict, model_path: str, image_folder: str | Path) -> list:
    # Загружаем модель
    model = YOLO(model_path)

    image_folder = Path(image_folder)
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    predictions = []
    for img_name in tqdm(images, desc="Выполняем предсказание на неразмеченных изображениях: "):
        img_path = image_folder / img_name
        result = model.predict(str(img_path))  # YOLO понимает str/Path, но явно приведём к str
        predictions.append((img_path, result))  # <--- сохраняем полный путь

    return predictions

# Функция для сохранения разметки боундбоксов на изображениях
def save_bounding_boxes(predictions: list, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    for img_path, result in tqdm(predictions, desc="Сохранение изображений для создания видео: "):
        img_path = Path(img_path)

        # Загружаем изображение
        img = Image.open(img_path).convert("RGB")

        # Получаем предсказанные боксы
        boxes = result[0].boxes.xywh.cpu().numpy()
        scores = result[0].boxes.conf.cpu().numpy()

        # Визуализируем боксы
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        ax.axis("off")

        for i, box in enumerate(boxes):
            if scores[i] < 0.5:  # порог уверенности
                continue

            x, y, w, h = box
            rect = patches.Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)

        # Имя файла без пути
        save_path = output_dir / img_path.name

        # Сохраняем изображение с разметкой
        fig.savefig(str(save_path), bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        saved_count += 1

    print(f"Сохранено изображений с баундбоксами: {saved_count} в {output_dir}")

# Функция для создания видео из изображений
def create_video_from_images(image_folder: str | Path, output_video_path: str, fps: int = 30):
    image_folder = Path(image_folder)

    # Получаем все изображения из папки
    images = sorted(
        [f for f in image_folder.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    )
    if not images:
        raise ValueError(f"В папке {image_folder} нет изображений для создания видео.")

    # Открываем первое изображение, чтобы получить размер
    first_img_path = str(images[0])
    frame = cv2.imread(first_img_path)
    if frame is None:
        raise ValueError(f"Не удалось прочитать изображение: {first_img_path}")
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Видео сохранено в: {output_video_path}")


# Основной процесс
def main(config_path: str = "configs/config.yaml"):
    # Загружаем конфиг
    cfg = load_config(config_path)
    # Получаем путь к последней обученной модели
    latest_model = get_latest_model(cfg)
    print(f"Последняя обученная модель: {latest_model}")

    # Папка с неразмеченными изображениями
    # 3) Корень YOLO-датасета
    PROJECT_ROOT = Path(__file__).resolve().parents[0]
    print(f"Корень проекта: {PROJECT_ROOT}")
    root_dir_cfg = Path(cfg["dataset"]["root_dir"])
    if root_dir_cfg.is_absolute():
        root_dir = root_dir_cfg
    else:
        root_dir = (PROJECT_ROOT / root_dir_cfg).resolve()
    unlabeled_dir = root_dir / cfg["dataset"]["unlabeled"]["images_dir"]
    print(f"Папка с неразмеченными изображениями: {unlabeled_dir}")

    # Выполняем предсказание на неразмеченных изображениях
    predictions = predict_on_unlabeled_images(cfg, latest_model, unlabeled_dir)

    # Сохраняем разметку на изображениях
    output_dir = Path("artefacts/annotated_images")
    save_bounding_boxes(predictions, output_dir)

    # Создаём видео с результатами
    output_video_path = "artefacts/annotated_video.mp4"
    create_video_from_images(output_dir, output_video_path)


if __name__ == "__main__":
    main("configs/config.yaml")

