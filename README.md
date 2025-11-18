# fish_spawning_detection

Detection of spawning fish in the streams of Russia, Yuzhno-Sakhalinsk

Source of idea [Curriculum Labeling: Revisiting Pseudo-Labeling for Semi-Supervised Learning.](https://arxiv.org/abs/2001.06001)
 and [Git](https://github.com/uvavision/Curriculum-Labeling/tree/main)

In this paper we revisit the idea of pseudo-labeling in the context of semi-supervised learning where a learning algorithm has access to a small set of labeled samples and a large set of unlabeled samples. Pseudo-labeling works by applying pseudo-labels to samples in the unlabeled set by using a model trained on the combination of the labeled samples and any previously pseudo-labeled samples, and iteratively repeating this process in a self-training cycle. Current methods seem to have abandoned this approach in favor of consistency regularization methods that train models under a combination of different styles of self-supervised losses on the unlabeled samples and standard supervised losses on the labeled samples. We empirically demonstrate that pseudo-labeling can in fact be competitive with the state-of-the-art, while being more resilient to out-of-distribution samples in the unlabeled set. We identify two key factors that allow pseudo-labeling to achieve such remarkable results (1) applying curriculum learning principles and (2) avoiding concept drift by restarting model parameters before each self-training cycle.

### или другими словами:



---

```
.
fish_spawning_detection/
├─ main_supervised.py           # 1-й запуск: базовое обучение только по разметке
├─ main_curriculum.py           # 2-й запуск: Curriculum Labeling (полу-наставник)
├─ process_predictions.py       # 3-й запуск: Разметка тестовых данных и ее визуализация
├─ configs/
│  └─ config.yaml               # пути и конфигурация основных модулей
└─ auxiliary/
   ├─ config_loader.py          # загрузка config.yaml
   ├─ coco_to_yolo.py           # конвертация COCO → YOLO + статистика по рыбам
   ├─ prepare_supervised.py     # подготовка данных + базовое обучение YOLO
   ├─ pseudo_labeling_yolo.py   # псевдо-разметка неразмеченных кадров YOLO
   └─ curriculum_yolo.py        # основной цикл Curriculum Labeling

   
```

## Логика Кода:

![Curriculum Labeling Pipeline](pics/Method.gif)

1. **Инициализация**: берём базовую модель YOLO (yolov8n.pt и т.п.).

2. **Шаг 0 (supervised)**: учим её **только на размеченном датасете** из instances_Train.json (только класс fish).

3. **Curriculum-итерации**:
   - каждый раз **переинициализируем** YOLO из базового чекпоинта; 
   - обучаем на **исходной разметке + всех псевдо-данных, полученных ранее**; 
   - прогоняем **неразмеченные кадры**, считаем «оценку кадра» по box-сcore (min/mean/max); 
   - по **перцентилю** выбираем порог, берём только «лёгкие» кадры с высокой уверенностью, сохраняем их псевдо-разметку в YOLO-формате; 
   - уменьшаем перцентиль → постепенно добавляем более сложные кадры (curriculum).

---

## Инструкция по запуску:

1. **Запуск** `main_supervised.py`:
- В `main_supervised.py` ты обучаешь модель на **размеченных данных** (только с метками, без псевдо-разметки).
- Когда обучение завершится (после 10 эпох, как ты указал в конфиге), ты получишь обученную модель, которая будет использовать только размеченные данные.
2. **Переключение на Curriculum Learning**:
- После завершения обучения на размеченных данных, можно переходить к Curriculum Learning. 
- Запуск main_curriculum.py необходим, чтобы начать дообучение модели на:
  - Размеченных данных (которые ты использовал в main_supervised.py). 
  - Псевдо-разметке, которую будет генерировать модель на каждой итерации.

3. **Запуск** `main_curriculum.py`:

После того как модель обучится на размеченных данных, ты можешь запускать `main_curriculum.py` для выполнения следующих шагов:

- **Curriculum Iterations**:
  - Модель будет переинициализирована и обучена сначала на самых уверенных псевдо-метках (которые модель сама будет генерировать, прогоняя неразмеченные кадры). 
  - Постепенно порог уверенности будет понижаться, и модель будет обучаться на всё более сложных примерах.

- **Основные этапы в** `main_curriculum.py`:
  - **Псевдо-метки** генерируются для неразмеченных данных. 
  - Псевдо-метки добавляются к данным для обучения. 
  - Модель дообучается на этих данных. 
  - Этот процесс повторяется несколько раз в рамках **Curriculum Learning**.

Запустить `main_curriculum.py` до завершения обучения на размеченных данных, 
то модель не будет обучена на основе базовых размеченных данных, 
и обучение будет происходить с нуля, что может не дать хороших результатов.

---

## Важные сообщения и их значения:

1. Отчёт о потере (box_loss, cls_loss, dfL_loss):

`Epoch 1/10 GPU_mem 0G box_loss 2.504 cls_loss 5.698 dfl_loss 1.776 Instances 28 Size 1280: 100% ━━━━━━━━`━━━━ 99/99 9.1s/it 15:03`

- **Epoch** 1/10 — это текущая эпоха (1 из 10). 
- **GPU_mem** — используемая память на GPU. 
- **box_loss**, **cls_loss**, **dfl_loss** — значения потерь для разных составляющих модели (обработка бокс-прогнозов, классификации, дифференцируемые потери). 
- **Instances** — количество объектов (или примеров) на текущую итерацию. 
- **Size** 1280:1280 — размер изображений (1280x1280 пикселей). 
- **100%** — это прогресс выполнения эпохи. 
- **9.1s/it** — время на одну итерацию. 
- **15:03** — общее время на текущую эпоху.

2. Точность:

`Class Images Instances Box(P R mAP50 mAP50-95): 39% ━━━━╸─────── 91/236 10.8s/it 16:18<26:12`

- **Class** — текущий класс, который тренируется (например, «fish»). 
- **Images** — количество изображений на этой итерации. 
- **Instances** — количество объектов на этой итерации. 
- **Box(P R)** — это Precision (точность) и Recall (полнота) для бокс-прогнозов. 
- **mAP50, mAP50-95** — это метрики для оценки качества модели (mean Average Precision), которые говорят о том, насколько точна модель на различных уровнях перекрытия коробок. 
- Эти значения обновляются по ходу тренировки.

3. Прогресс тренировки:

`optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...`

`optimizer: AdamW(lr=0.002, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)`

- Информация о том, какой оптимизатор используется (например, AdamW) и какие параметры оптимизации автоматически выбраны моделью.


