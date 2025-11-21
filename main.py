# main.py
import subprocess

from auxiliary.prepare_supervised import train_supervised
from auxiliary.curriculum_yolo import curriculum_training
from add_ons.collect_yolo_metrics import setup_pandas_options

setup_pandas_options()

if __name__ == "__main__":

    """
    Сначала модель обучается на размеченных данных 
    (только с метками, без псевдо-разметки).
    """
    train_supervised("configs/config.yaml")  # first launch
    """
    Потом запускаются итерации Curriculum Labeling:
        - каждый раз заново инициализирует YOLO;
        - обучает на разметке + прошлых pseudo;
        - псевдо-размечает неразмеченные кадры;
        - добавляет всё более сложные примеры по перцентилям.
    """
    curriculum_training("configs/config.yaml")  # second launch

    # Collect & plots metrics
    subprocess.run(["python", "add_ons/collect_yolo_metrics.py"])
