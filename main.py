# main.py
from auxiliary.prepare_supervised import train_supervised
from auxiliary.curriculum_yolo import curriculum_training
from add_ons.collect_yolo_metrics import (
    load_results,
    save_merged_results,
    plot_yolo_like_report,
    setup_pandas_options,
    plot_yolo_like_report_plotly)

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
    df_all = load_results()
    save_merged_results(df_all)
    plot_yolo_like_report(df_all)
    plot_yolo_like_report_plotly(df_all)
