# main_supervised.py
# first launch
"""
запустит итерации Curriculum Labeling:
    - каждый раз заново инициализирует YOLO;
    - обучает на разметке + прошлых pseudo;
    - псевдо-размечает неразмеченные кадры;
    - добавляет всё более сложные примеры по перцентилям.
"""
from auxiliary.prepare_supervised import train_supervised

if __name__ == "__main__":
    train_supervised("configs/config.yaml")
