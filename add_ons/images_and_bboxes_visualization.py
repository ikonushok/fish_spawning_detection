#!/usr/bin/env python3
import os
import json
from collections import defaultdict
from typing import Dict, Tuple, Optional, Any, List

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from tqdm import tqdm


# ====================== ЛОГИЧЕСКИЕ ФУНКЦИИ ======================

def load_coco_annotations(annotations_dir: str,
                          annot_file_name: str) -> Tuple[Dict[int, dict], Dict[int, str], Dict[int, list]]:
    """
    Загружает COCO-аннотации и возвращает:
    - images: {image_id -> image_info}
    - categories: {category_id -> category_name}
    - anns_by_image: {image_id -> [annotations]}
    """
    ann_path = os.path.join(annotations_dir, annot_file_name)
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Не найден файл аннотаций: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {c["id"]: c["name"] for c in coco["categories"]}

    anns_by_image: Dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    return images, categories, anns_by_image


def prepare_font(font_size: int = 16) -> ImageFont.FreeTypeFont:
    """
    Загружает шрифт. Если указанный ttf не найден — берёт дефолтный.
    """
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    return font


def draw_bboxes_on_image(img: Image.Image,
                         anns_for_image: List[dict],
                         categories: Dict[int, str],
                         font: ImageFont.FreeTypeFont) -> Image.Image:
    """
    Рисует баундбоксы и подписи на изображении (PIL.Image) и возвращает его же.
    """
    draw = ImageDraw.Draw(img)

    for ann in anns_for_image:
        x, y, w, h = ann["bbox"]   # COCO: [x, y, width, height]
        x2, y2 = x + w, y + h

        cat_name = categories.get(ann["category_id"], str(ann["category_id"]))

        # прямоугольник
        draw.rectangle([x, y, x2, y2], outline="red", width=2)

        # подпись категории поверх бокса
        text = cat_name
        # bbox текста: (left, top, right, bottom)
        tb_left, tb_top, tb_right, tb_bottom = draw.textbbox((0, 0), text, font=font)
        tw = tb_right - tb_left
        th = tb_bottom - tb_top

        text_x = x
        text_y = y - th
        if text_y < 0:
            text_y = 0

        draw.rectangle([text_x, text_y, text_x + tw, text_y + th], fill="red")
        draw.text((text_x, text_y), text, fill="yellow", font=font)

    return img


def init_video_writer(out_video_path: str,
                      frame_size: Tuple[int, int],
                      fps: int) -> cv2.VideoWriter:
    """
    Инициализирует cv2.VideoWriter для записи видео.
    frame_size: (width, height)
    """
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, frame_size)
    print(f"\n[VIDEO] Инициализирована запись видео:\t{out_video_path} {frame_size[0]}x{frame_size[1]} @ {fps} fps)")
    return writer


def write_frame_to_video(video_writer: cv2.VideoWriter,
                         img: Image.Image,
                         target_size: Tuple[int, int]):
    """
    Конвертирует PIL.Image в формат OpenCV (BGR), подгоняет размер и пишет кадр в видео.
    """
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if (frame.shape[1], frame.shape[0]) != target_size:
        frame = cv2.resize(frame, target_size)
    video_writer.write(frame)


def process_dataset(
    annotations_dir: str,
    annot_file_name: str,
    images_dir: str,
    out_dir_images: str,
    out_video_path: str,
    save_images: bool,
    save_video: bool,
    video_fps: int
):
    """
    Основной цикл:
    - читает COCO
    - проходит по всем изображениям
    - рисует боксы
    - при необходимости сохраняет картинки и видео
    """
    if not save_images and not save_video:
        print("Оба флажка save_images и save_video = False — делать нечего.")
        return

    if save_images:
        os.makedirs(out_dir_images, exist_ok=True)

    images, categories, anns_by_image = load_coco_annotations(annotations_dir, annot_file_name)
    font = prepare_font(font_size=16)

    video_writer: Optional[cv2.VideoWriter] = None
    video_size: Optional[Tuple[int, int]] = None

    # Чтобы видео шло в правильном порядке, отсортируем по имени файла
    img_items = sorted(images.items(), key=lambda kv: kv[1]["file_name"])

    for img_id, img_info in tqdm(img_items):
        file_name = img_info["file_name"]
        in_path = os.path.join(images_dir, file_name)

        if save_images:
            out_img_path = os.path.join(out_dir_images, file_name)

        if not os.path.exists(in_path):
            print(f"[WARN] Изображение не найдено: {in_path}")
            continue

        img = Image.open(in_path).convert("RGB")
        anns_for_image = anns_by_image.get(img_id, [])

        img = draw_bboxes_on_image(img, anns_for_image, categories, font)

        # --- Сохранение изображения ---
        if save_images:
            os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
            img.save(out_img_path)
            # print(f"[IMG] {file_name} -> {out_img_path}")

        # --- Добавление кадра в видео ---
        if save_video:
            if video_writer is None:
                width, height = img.size  # PIL: (width, height)
                video_size = (width, height)
                video_writer = init_video_writer(out_video_path, video_size, video_fps)

            write_frame_to_video(video_writer, img, video_size)
            # print(f"[VID] Добавлен кадр {file_name}")

    if video_writer is not None:
        video_writer.release()
        print(f"[VIDEO] Видео сохранено: {out_video_path}")

    print("Готово.")


def main(save_images: bool,
         save_video: bool,
         annotations_dir: str,
         annot_file_name: str,
         images_dir: str,
         out_dir_images: str,
         out_video_path: str,
         video_fps: int):
    """
    Обёртка над process_dataset. Можно вызвать и из другого кода.
    """
    process_dataset(
        annotations_dir=annotations_dir,
        annot_file_name=annot_file_name,
        images_dir=images_dir,
        out_dir_images=out_dir_images,
        out_video_path=out_video_path,
        save_images=save_images,
        save_video=save_video,
        video_fps=video_fps,
    )


# ====================== ТОЧКА ВХОДА ======================

if __name__ == "__main__":
    # ---------- ФЛАЖКИ ----------
    SAVE_IMAGES = False   # сохранять изображения с баундбоксами
    SAVE_VIDEO  = True   # собирать видео из изображений с баундбоксами

    # ---------- ПУТИ ----------
    source_path = "../dataset"
    stream_path = "job_196_dataset_2025_09_03_03_38_08_coco 1"
    ANNOTATIONS_DIR = f"{source_path}/{stream_path}/annotations"
    IMAGES_DIR      = f"{source_path}/{stream_path}/images/default"
    OUT_DIR_IMAGES  = f"{source_path}/{stream_path}/images_bbox"
    OUT_VIDEO_PATH  = f"{source_path}/{stream_path}/bboxes_video.mp4"

    # ---------- ПРОЧЕЕ ----------
    ANNOT_FILE_NAME = "instances_default.json"
    VIDEO_FPS = 25

    # запуск
    main(
        save_images=SAVE_IMAGES,
        save_video=SAVE_VIDEO,
        annotations_dir=ANNOTATIONS_DIR,
        annot_file_name=ANNOT_FILE_NAME,
        images_dir=IMAGES_DIR,
        out_dir_images=OUT_DIR_IMAGES,
        out_video_path=OUT_VIDEO_PATH,
        video_fps=VIDEO_FPS,
    )
