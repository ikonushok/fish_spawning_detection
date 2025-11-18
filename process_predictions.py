import os
import cv2
import yaml
from pathlib import Path
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
def predict_on_unlabeled_images(cfg: dict, model_path: str, image_folder: str) -> list:
    # Загружаем модель
    model = YOLO(model_path)

    # Получаем список изображений для предсказания
    images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    predictions = []
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        result = model.predict(img_path)  # Выполняем предсказание
        predictions.append((img_name, result))

    return predictions


# Функция для сохранения разметки боундбоксов на изображениях
def save_bounding_boxes(predictions: list, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Создаём папку для сохранённых изображений

    for img_name, result in predictions:
        # Получаем предсказанные боксы
        boxes = result[0].boxes.xywh.cpu().numpy()
        scores = result[0].boxes.conf.cpu().numpy()

        # Загружаем изображение для добавления боундбоксов
        img = Image.open(img_name)

        # Визуализируем боксы
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        for i, box in enumerate(boxes):
            if scores[i] < 0.5:  # Применяем порог уверенности
                continue

            x, y, w, h = box
            rect = patches.Rectangle(
                (x - w / 2, y - h / 2), w, h, linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

        # Сохраняем изображение с разметкой
        save_path = output_dir / img_name
        plt.savefig(save_path)
        plt.close(fig)


# Функция для создания видео из изображений
def create_video_from_images(image_folder: str, output_video_path: str, fps: int = 30):
    # Получаем все изображения из папки
    images = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if not images:
        raise ValueError("В папке нет изображений для создания видео.")

    # Открываем первое изображение, чтобы получить размер
    first_img_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_img_path)
    height, width, _ = frame.shape

    # Создаём объект для записи видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Добавляем изображения в видео
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    video_writer.release()  # Закрываем видеофайл


# Основной процесс
def main(config_path: str = "config.yaml"):
    # Загружаем конфиг
    cfg = load_config(config_path)

    # Получаем путь к последней обученной модели
    latest_model = get_latest_model(cfg)
    print(f"Последняя обученная модель: {latest_model}")

    # Папка с неразмеченными изображениями
    unlabeled_folder = cfg["dataset"]["unlabeled"]["images_dir"]

    # Выполняем предсказание на неразмеченных изображениях
    predictions = predict_on_unlabeled_images(cfg, latest_model, unlabeled_folder)

    # Сохраняем разметку на изображениях
    output_dir = "artefacts/annotated_images"
    save_bounding_boxes(predictions, output_dir)

    # Создаём видео с результатами
    output_video_path = "artefacts/annotated_video.mp4"
    create_video_from_images(output_dir, output_video_path)
    print(f"Видео создано: {output_video_path}")


if __name__ == "__main__":
    main("config.yaml")

