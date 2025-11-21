
from PIL import Image, ImageFilter
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

# Убираем блики с помощью гауссового размытия
def remove_glare(image: Image) -> Image:
    """
    Убираем блики с изображения с помощью гауссового размытия.
    :param image: Входное изображение (PIL Image)
    :return: Изображение с уменьшенным бликованием (PIL Image)
    """
    return image.filter(ImageFilter.GaussianBlur(radius=2))

# Убираем блики с использованием медианного фильтра OpenCV
def remove_glare_opencv(image: np.array) -> np.array:
    """
    Убираем блики с изображения с помощью медианного фильтра.
    :param image: Входное изображение (NumPy Array)
    :return: Изображение с уменьшенным бликованием (NumPy Array)
    """
    return cv2.medianBlur(image, 5)

# Основная функция для предобработки изображения
def preprocess_image(image_path: Path) -> Image:
    """
    Загружаем изображение, удаляем блики и применяем другие предобработки.
    :param image_path: Путь к изображению
    :return: Обработанное изображение
    """
    # Открываем изображение
    image = Image.open(image_path)

    # Убираем блики с помощью гауссового размытия
    image = remove_glare(image)

    # Дополнительная обработка: например, изменение размера
    image = image.resize((640, 640))

    return image

# Предобработка с использованием OpenCV (если нужно)
def preprocess_image_opencv(image_path: Path) -> np.array:
    """
    Предобработка изображения с использованием OpenCV
    :param image_path: Путь к изображению
    :return: Обработанное изображение
    """
    # Читаем изображение с помощью OpenCV
    image = cv2.imread(str(image_path))

    # Убираем блики с помощью медианного фильтра
    image = remove_glare_opencv(image)

    # Преобразуем изображение в RGB для совместимости с YOLO
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image



class FishDataset(Dataset):
    def __init__(self, image_paths, labels_paths, transform=None):
        """
        Инициализация датасета.
        :param image_paths: Список путей к изображениям
        :param labels_paths: Список путей к меткам
        :param transform: Функция для преобразования изображений
        """
        self.image_paths = image_paths
        self.labels_paths = labels_paths
        self.transform = transform

    def __len__(self):
        """Возвращаем количество изображений в датасете."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Получаем изображение и метку для одного элемента."""
        image = Image.open(self.image_paths[idx])
        label = self.labels_paths[idx]

        # Убираем блики на изображении
        image = preprocess_image(self.image_paths[idx])

        # Применяем дополнительные аугментации, если нужно
        if self.transform:
            image = self.transform(image)

        # Преобразуем изображение в формат, который YOLO ожидает
        image = np.array(image)  # Преобразуем в numpy массив
        image = torch.from_numpy(image).float()  # Преобразуем в Tensor

        return image, label
