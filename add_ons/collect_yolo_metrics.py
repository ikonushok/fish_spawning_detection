import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def setup_pandas_options():
    """Настройка глобальных опций pandas для отображения."""
    pd.set_option("display.precision", 3)
    pd.set_option("expand_frame_repr", False)


setup_pandas_options()

# Пути к результатам
SUPERVISED_DIR = Path("../artefacts/runs/supervised_yolo/baseline/results.csv")
CURRICULUM_DIR = Path("../artefacts/runs/curriculum_yolo/")
REPORT_DIR = Path("../artefacts/runs")                 # куда сохраняем общий отчёт
REPORT_NAME_IMG = "curriculum_report.png"             # имя файла с графиками
REPORT_NAME_CSV = "curriculum_results_merged.csv"     # имя объединённого CSV


def load_results() -> pd.DataFrame:
    """Считываем supervised и все итерации curriculum в один DataFrame."""

    all_dfs = []

    # --- Supervised (этап -1) ---
    if SUPERVISED_DIR.exists():
        df_sup = pd.read_csv(SUPERVISED_DIR)
        df_sup["iter"] = -1
        df_sup["phase"] = "supervised"
        all_dfs.append(df_sup)
    else:
        raise FileNotFoundError(f"Не найден supervised results: {SUPERVISED_DIR}")

    # --- Curriculum iter_* ---
    for iter_dir in sorted(CURRICULUM_DIR.glob("iter_*")):
        iter_idx = int(iter_dir.name.split("_")[1])
        csv_path = iter_dir / "results.csv"

        if not csv_path.exists():
            print(f"[WARN] Пропускаю {csv_path} — файла нет")
            continue

        df = pd.read_csv(csv_path)
        df["iter"] = iter_idx
        df["phase"] = f"curriculum_{iter_idx}"
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("Не найдено ни одного results.csv")

    df_all = pd.concat(all_dfs, ignore_index=True)

    # сортировка по итерации и эпохе
    df_all = df_all.sort_values(["iter", "epoch"]).reset_index(drop=True)

    # посчитаем глобальный шаг (эпохи всех этапов подряд)
    global_step = []
    offset = 0
    for _, df_phase in df_all.groupby("iter", sort=True):
        epochs = df_phase["epoch"].values
        global_step.extend(epochs + offset)
        offset += epochs.max() + 1  # следующий этап начинается после последней эпохи

    df_all["global_epoch"] = global_step

    return df_all


def save_merged_results(df_all: pd.DataFrame,
                        save_dir: Path = REPORT_DIR,
                        save_name: str = REPORT_NAME_CSV):
    """Сохраняем объединённый DataFrame в CSV в папку отчётов."""
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_name
    df_all.to_csv(save_path, index=False)
    print(f"[INFO] Итоговый CSV с результатами сохранён в: {save_path.resolve()}")


def plot_yolo_like_report(
    df_all: pd.DataFrame,
    use_global_epoch: bool = True,
    save_dir: Path = REPORT_DIR,
    save_name: str = REPORT_NAME_IMG,
):
    """
    Рисует отчёт в форме, похожей на yolo results.png:
    2x5 subplot'ов с основными метриками.
    Легенда внизу, общий заголовок вверху.
    Также сохраняет картинку в save_dir / save_name.
    """

    x_col = "global_epoch" if use_global_epoch else "epoch"

    metrics = [
        ("train/box_loss",       "train/box_loss"),
        ("train/cls_loss",       "train/cls_loss"),
        ("train/dfl_loss",       "train/dfl_loss"),
        ("metrics/precision(B)", "metrics/precision(B)"),
        ("metrics/recall(B)",    "metrics/recall(B)"),
        ("val/box_loss",         "val/box_loss"),
        ("val/cls_loss",         "val/cls_loss"),
        ("val/dfl_loss",         "val/dfl_loss"),
        ("metrics/mAP50(B)",     "metrics/mAP50(B)"),
        ("metrics/mAP50-95(B)",  "metrics/mAP50-95(B)"),
    ]

    # УВЕЛИЧИЛ высоту: было figsize=(20, 8)
    fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharex=True)
    axes = axes.ravel()

    phases = df_all["phase"].unique()

    for ax, (col, title) in zip(axes, metrics):
        if col not in df_all.columns:
            ax.set_title(f"{title}\n[нет в CSV]")
            ax.axis("off")
            continue

        for phase in phases:
            df_phase = df_all[df_all["phase"] == phase]
            ax.plot(df_phase[x_col], df_phase[col], marker=".", label=phase)

        ax.set_title(title)
        ax.grid(True)

    # Подписи осей только снизу/слева
    for i in range(5):
        axes[i + 5].set_xlabel("global epoch" if use_global_epoch else "epoch")
    axes[0].set_ylabel("value")
    axes[5].set_ylabel("value")

    # Общий заголовок
    fig.suptitle("Supervised + Curriculum YOLO training report", fontsize=16, y=0.97)

    # Общая легенда внизу – чуть сильнее опускаем её
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 5),
        bbox_to_anchor=(0.5, -0.03),  # было -0.02
    )

    # Больше воздуха снизу под легенду
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    # --- Сохранение графика ---
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_name
    fig.savefig(save_path, dpi=200, bbox_inches="tight")  # добавил bbox_inches="tight"
    print(f"[INFO] Отчёт с графиками сохранён в: {save_path.resolve()}")

    plt.show()


def plot_yolo_like_report_plotly(df_all, use_global_epoch: bool = True):
    """
    Интерактивный отчёт в стиле yolo results с помощью plotly.
    2x5 subplot'ов, легенда внизу, общий заголовок наверху.

    :param df_all: DataFrame с колонками 'phase', 'epoch' и 'global_epoch' + метриками
    :param use_global_epoch: если True — по оси X используем 'global_epoch', иначе 'epoch'
    :return: plotly.graph_objects.Figure
    """

    x_col = "global_epoch" if use_global_epoch else "epoch"

    metrics = [
        ("train/box_loss",       "train/box_loss"),
        ("train/cls_loss",       "train/cls_loss"),
        ("train/dfl_loss",       "train/dfl_loss"),
        ("metrics/precision(B)", "metrics/precision(B)"),
        ("metrics/recall(B)",    "metrics/recall(B)"),
        ("val/box_loss",         "val/box_loss"),
        ("val/cls_loss",         "val/cls_loss"),
        ("val/dfl_loss",         "val/dfl_loss"),
        ("metrics/mAP50(B)",     "metrics/mAP50(B)"),
        ("metrics/mAP50-95(B)",  "metrics/mAP50-95(B)"),
    ]

    # создаём сетку 2x5 с заголовками подграфиков
    fig = make_subplots(
        rows=2,
        cols=5,
        shared_xaxes=True,
        subplot_titles=[title for _, title in metrics],
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
    )

    phases = df_all["phase"].unique()

    for i, (col, title) in enumerate(metrics):
        row = i // 5 + 1
        col_idx = i % 5 + 1

        if col not in df_all.columns:
            # просто оставим пустой график, без трасс
            continue

        for phase in phases:
            df_phase = df_all[df_all["phase"] == phase]

            fig.add_trace(
                go.Scatter(
                    x=df_phase[x_col],
                    y=df_phase[col],
                    mode="lines+markers",
                    name=phase,
                    showlegend=(i == 0),  # легенду показываем только один раз
                ),
                row=row,
                col=col_idx,
            )

    # подписи осей X только внизу
    xaxis_title = "global epoch" if use_global_epoch else "epoch"
    for j in range(1, 6):
        fig.update_xaxes(title_text=xaxis_title, row=2, col=j)

    # можно задать подписи Y для левого столбца
    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_yaxes(title_text="value", row=2, col=1)

    # общий заголовок и легенда внизу
    fig.update_layout(
        title_text="Supervised + Curriculum YOLO training report (Plotly)",
        title_x=0.5,
        height=900,
        width=1600,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=80, b=80, l=60, r=20),
    )

    fig.show(renderer="browser")


if __name__ == "__main__":
    df_all = load_results()
    print(df_all.head())

    # сохраняем объединённый CSV
    save_merged_results(df_all)

    # рисуем и сохраняем картинку-отчёт
    plot_yolo_like_report(df_all)
    plot_yolo_like_report_plotly(df_all)
