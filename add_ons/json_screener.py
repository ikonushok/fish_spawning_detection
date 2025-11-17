#!/usr/bin/env python3
import os
import json
import sys
from typing import Any
import webbrowser
from pathlib import Path


def summarize_classic_one_line(obj: Any, max_items: int = 5) -> str:
    """
    Возвращает краткое однострочное описание dict/list в стиле:
    dict (3 keys): 'name': str, 'id': int, 'url': str
    """
    if isinstance(obj, dict):
        items = list(obj.items())
        total = len(items)
        items = items[:max_items]
        parts = [f"'{k}': {type(v).__name__}" for k, v in items]
        suffix = ""
        if total > max_items:
            suffix = f", ... (+{total - max_items} more)"
        return f"dict ({total} keys): " + ", ".join(parts) + suffix

    elif isinstance(obj, list):
        items = list(enumerate(obj))
        total = len(items)
        items = items[:max_items]
        parts = [f"[{i}]: {type(v).__name__}" for i, v in items]
        suffix = ""
        if total > max_items:
            suffix = f", ... (+{total - max_items} more)"
        return f"list ({total} items): " + ", ".join(parts) + suffix

    else:
        return f"{type(obj).__name__}: {repr(obj)}"


def print_structure_tree(
    obj: Any,
    prefix: str = "",
    max_depth: int = 3,
    max_items: int = 5,   # теперь влияет только на глубину, но не на ключи
    level: int = 0,
):
    """
    Улучшенный tree-вывод:
    - все ключи dict выводятся полностью
    - у списков показывается только первый элемент
    - затем пишется "... (N more items)"
    """

    def print_child(name: str, value: Any, is_last: bool):
        branch = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")
        type_name = type(value).__name__

        print(prefix + branch + f"{name}: {type_name}")

        if isinstance(value, (dict, list)):
            print_structure_tree(
                value,
                prefix=child_prefix,
                max_depth=max_depth,
                max_items=max_items,
                level=level + 1,
            )

    # --------------------------
    # dict
    # --------------------------
    if isinstance(obj, dict):
        items = list(obj.items())  # НЕ обрезаем max_items
        total = len(items)

        for idx, (k, v) in enumerate(items):
            print_child(f"'{k}'", v, is_last=(idx == len(items) - 1))

        return

    # --------------------------
    # list
    # --------------------------
    elif isinstance(obj, list):
        total = len(obj)

        if total == 0:
            print(prefix + "└── [] (empty)")
            return

        # показываем только первый элемент списка
        i, v = 0, obj[0]
        print_child(f"[0]", v, is_last=True)

        # выводим число оставшихся
        if total > 1:
            print(prefix + f"... ({total - 1} more items)")

        return

    # --------------------------
    # value
    # --------------------------
    else:
        preview = repr(obj)
        preview = preview if len(preview) <= 80 else preview[:77] + "..."
        print(prefix + f"value: {preview}")


def inspect_json_file(
    path: str,
    base_indent: int,
    max_depth: int,
    max_items: int,
    tree_style: bool,
):
    print("=" * 80)
    print(f"Файл: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Ошибка чтения JSON: {e}")
        return

    print("Тип корневого объекта:", type(data).__name__)

    if tree_style:
        # --- вывод в консоль деревом ---
        prefix = "  " * base_indent
        print_structure_tree(
            data,
            prefix=prefix,
            max_depth=max_depth,
            max_items=max_items,
            level=0,
        )

    else:
        # --- ОТКРЫТЬ JSON В БРАУЗЕРЕ ---
        from pathlib import Path  # можно здесь или сверху
        uri = Path(path).resolve().as_uri()
        print(f"Открываю в браузере: {uri}")
        webbrowser.open(uri)


def inspect_folder(
    root_dir: str,
    limit_files: int = 5,
    base_indent: int = 1,
    max_depth: int = 3,
    max_items: int = 5,
    tree_style: bool = False,
):
    """
    Ищет JSON-файлы в подпапках и печатает структуру.

    Параметры:
      base_indent — отступы для classic-вывода
      max_depth — глубина рекурсии
      max_items — макс. число элементов на уровне
      tree_style — использовать ASCII-дерево вместо classic-формата
    """
    count = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".json"):
                full_path = os.path.join(dirpath, filename)
                inspect_json_file(
                    full_path,
                    base_indent=base_indent,
                    max_depth=max_depth,
                    max_items=max_items,
                    tree_style=tree_style,
                )
                count += 1
                if count >= limit_files:
                    print(f"\nПоказано {count} файлов (limit_files={limit_files}).")
                    return

    if count == 0:
        print("JSON-файлы не найдены.")


if __name__ == "__main__":
    # Папка передаётся первым аргументом, по умолчанию – текущая
    root = '../dataset/job_175_dataset_2025_09_08_02_20_40_coco 1'
    inspect_folder(
        root_dir=root,
        limit_files=2,  # просмотреть больше файлов
        base_indent=1,  # параметр, который отвечает за красивую иерархическую печать структуры JSON, делая её читаемой
        max_depth=3,    # увеличить глубину показа структуры
        max_items=0,    # показать больше ключей/элементов
        tree_style=False,   # изменить стиль вывода собранной информации
    )
