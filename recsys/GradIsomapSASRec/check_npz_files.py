#!/usr/bin/env python3
"""
Сканирование папки с .npz файлами на целостность.

Использование:
    python check_npz.py <путь_до_папки>
    python check_npz.py <путь_до_папки> --recursive
"""

import argparse
import sys
import zipfile
from pathlib import Path


def is_valid_npz(path: Path) -> bool:
    """Проверяет, что файл — валидный zip-контейнер (NPZ)."""
    if not path.is_file():
        return False
    if path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            # проверим, что центральное оглавление читается и архиву можно доверять
            bad = zf.testzip()
            if bad is not None:
                return False
    except (zipfile.BadZipFile, OSError, EOFError):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка .npz файлов на целостность")
    parser.add_argument("folder", type=Path, help="Путь до папки с .npz файлами")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Рекурсивно обходить подпапки")
    args = parser.parse_args()

    folder: Path = args.folder
    if not folder.is_dir():
        print(f"Не папка: {folder}", file=sys.stderr)
        return 2

    pattern = "**/*.npz" if args.recursive else "*.npz"
    files = sorted(folder.glob(pattern))

    if not files:
        print(f"В папке нет .npz файлов: {folder}", file=sys.stderr)
        return 1

    problems = 0
    for f in files:
        if not is_valid_npz(f):
            print(f)
            problems += 1

    print(f"\nПроверено: {len(files)}, проблемных: {problems}", file=sys.stderr)
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
    