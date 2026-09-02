from collections import defaultdict
import pandas as pd
import numpy as np
import os
import gzip
import json
import requests
from tqdm import tqdm


def prepare_sequences(df):

    user_ids = df["userId"].unique()
    movie_ids = df["movieId"].unique()
    user2idx = {u: i for i, u in enumerate(sorted(user_ids))}
    movie2idx = {m: i for i, m in enumerate(sorted(movie_ids))}

    df["user_idx"] = df["userId"].map(user2idx)
    df["movie_idx"] = df["movieId"].map(movie2idx)
    df = df.sort_values(["user_idx", "timestamp"])

    user2seq = defaultdict(list)
    for row in df.itertuples():
        user2seq[row.user_idx].append((row.movie_idx, row.timestamp, row.rating))

    return df, user2seq


# Официальный источник: https://amazon-reviews-2023.github.io/
AMAZON_BOOKS_REVIEWS_URL = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/"
    "raw/review_categories/Books.jsonl.gz"
)


def download_file(url: str, dest_path: str, chunk_size: int = 1 << 20) -> None:
    """Скачивает файл по URL с прогресс-баром. Пропускает если уже есть."""
    if os.path.exists(dest_path):
        print(f"[Download] Уже скачан: {dest_path}")
        return

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    print(f"[Download] Скачиваем {url}")
    print(f"           → {dest_path}")

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="Books.jsonl.gz"
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))

    print(f"[Download] Готово: {dest_path}")


def load_amazon_books(
    subset_path: str = "recsys/GradIsomapCF_movielens/data/amazon_books/amazon_books_5000u_10000i.parquet",
    **kwargs,                    # игнорируем лишние параметры
) -> pd.DataFrame:
    """
    Загружает готовую подвыборку Amazon Books.
    Файл создаётся один раз в Google Colab (см. notebook).
    """
    if not os.path.exists(subset_path):
        data_path = "recsys/GradIsomapCF_movielens/data"

        # Проверяем, существует ли папка data
        if os.path.exists(data_path):
            print(f"📁 Содержимое папки '{data_path}':")
            print("=" * 50)
            
            # Рекурсивный обход всех папок и файлов
            for root, dirs, files in os.walk(data_path):
                # Уровень вложенности (для отступов)
                level = root.replace(data_path, '').count(os.sep)
                indent = ' ' * 2 * level
                
                # Выводим текущую папку
                print(f"{indent}📁 {os.path.basename(root)}/")
                
                # Выводим файлы в папке
                sub_indent = ' ' * 2 * (level + 1)
                for file in files:
                    file_size = os.path.getsize(os.path.join(root, file))
                    # Форматируем размер файла
                    if file_size < 1024:
                        size_str = f"{file_size} B"
                    elif file_size < 1024 * 1024:
                        size_str = f"{file_size / 1024:.2f} KB"
                    else:
                        size_str = f"{file_size / (1024 * 1024):.2f} MB"
                    
                    print(f"{sub_indent}📄 {file} ({size_str})")
                
                # Если папка пустая
                if not files and not dirs:
                    print(f"{sub_indent}⚠️  Папка пуста")
                
                print()  # Пустая строка для разделения
            
            # Статистика
            total_files = sum(len(files) for _, _, files in os.walk(data_path))
            total_dirs = sum(len(dirs) for _, dirs, _ in os.walk(data_path))
            print("=" * 50)
            print(f"📊 Итого: {total_dirs} папок, {total_files} файлов")
            
        else:
            print(f"❌ Папка '{data_path}' не существует!")
            print(f"Текущая директория: {os.getcwd()}")
            print(f"Содержимое текущей директории: {os.listdir('.')}")
        raise FileNotFoundError(
            f"Файл подвыборки не найден: {subset_path}\n"
            f"Создайте его в Google Colab с помощью скрипта подготовки.\n"
            f"Ожидаемые колонки: userId, movieId, rating, timestamp"
        )

    print(f"[Amazon Books] Загружаем подвыборку: {subset_path}")
    ext = os.path.splitext(subset_path)[1].lower()

    if ext == ".parquet":
        df = pd.read_parquet(subset_path)
    elif ext == ".csv":
        df = pd.read_csv(subset_path)
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    # Проверяем схему
    required = {"userId", "movieId", "rating", "timestamp"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"В файле отсутствуют колонки: {missing}")

    df["rating"]    = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    df = df[df["rating"] > 0].dropna(subset=list(required))

    print(f"  Пользователей: {df['userId'].nunique():,}")
    print(f"  Книг:          {df['movieId'].nunique():,}")
    print(f"  Записей:       {len(df):,}")

    return df


def prepare_sequences_amazon(df: pd.DataFrame):
    """
    Аналог prepare_sequences для Amazon Books.
    userId — строка (хэш), movieId — строка (asin).
    Маппим оба в contiguous int-индексы.
    """
    # Сортируем строковые ID для детерминированного маппинга
    user_ids  = sorted(df["userId"].unique())
    item_ids  = sorted(df["movieId"].unique())

    user2idx  = {u: i for i, u in enumerate(user_ids)}
    item2idx  = {m: i for i, m in enumerate(item_ids)}

    df = df.copy()
    df["user_idx"]  = df["userId"].map(user2idx)
    df["movie_idx"] = df["movieId"].map(item2idx)
    df = df.sort_values(["user_idx", "timestamp"])

    user2seq = defaultdict(list)
    for row in df.itertuples():
        user2seq[row.user_idx].append(
            (row.movie_idx, row.timestamp, row.rating)
        )

    return df, user2seq, user2idx, item2idx


def subsample_users_items(df_mapped,
                          max_users=500,
                          max_movies=1000,
                          min_seq_len=5):

    user_counts = df_mapped.groupby("user_idx").size()
    good_users = user_counts[user_counts >= min_seq_len].index.values
    if len(good_users) > max_users:
        good_users = user_counts.loc[good_users].sort_values(ascending=False).head(max_users).index.values

    df_sub = df_mapped[df_mapped["user_idx"].isin(good_users)].copy()

    movie_counts = df_sub.groupby("movie_idx").size()
    good_movies = movie_counts.index.values
    if len(good_movies) > max_movies:
        good_movies = movie_counts.sort_values(ascending=False).head(max_movies).index.values

    df_sub = df_sub[df_sub["movie_idx"].isin(good_movies)].copy()

    user_counts2 = df_sub.groupby("user_idx").size()
    good_users2 = user_counts2[user_counts2 >= min_seq_len].index.values
    df_sub = df_sub[df_sub["user_idx"].isin(good_users2)].copy()

    old_user_ids = sorted(df_sub["user_idx"].unique())
    old_movie_ids = sorted(df_sub["movie_idx"].unique())
    user_remap = {u: i for i, u in enumerate(old_user_ids)}
    movie_remap = {m: i for i, m in enumerate(old_movie_ids)}

    df_sub["user_idx"] = df_sub["user_idx"].map(user_remap)
    df_sub["movie_idx"] = df_sub["movie_idx"].map(movie_remap)

    num_users = df_sub["user_idx"].nunique()
    num_movies = df_sub["movie_idx"].nunique()

    df_sub = df_sub.sort_values(["user_idx", "timestamp"])
    user2seq_sub = defaultdict(list)
    for row in df_sub.itertuples():
        user2seq_sub[row.user_idx].append((row.movie_idx, row.timestamp, row.rating))

    print(f"  После подвыборки: пользователей={num_users}, фильмов={num_movies}, записей={len(df_sub)}")
    return df_sub, user2seq_sub, num_users, num_movies


def train_test_split_next_item(user2seq, min_len=2):

    train_events = []
    test_next = []

    for u, seq in user2seq.items():
        if len(seq) < min_len:
            continue
        movies = [x[0] for x in seq]
        ratings = [x[2] for x in seq]

        target_movie = movies[-1]
        last_train_movie = movies[-2]

        for m, r in zip(movies[:-1], ratings[:-1]):
            train_events.append((u, m, r))
        test_next.append((u, last_train_movie, target_movie))

    train_events = np.array(train_events, dtype=np.int64)
    test_next = np.array(test_next, dtype=np.int64)
    return train_events, test_next


def train_val_test_split_next_item(user2seq, min_len=3):

    train_events = []
    val_next = []
    test_next = []

    for u, seq in user2seq.items():
        if len(seq) < min_len:
            continue

        movies = [x[0] for x in seq]
        ratings = [x[2] for x in seq]

        # последний фильм -> test
        test_movie = movies[-1]
        last_train_for_test = movies[-2]

        # предпоследний фильм -> val
        val_movie = movies[-2]
        last_train_for_val = movies[-3] if len(movies) >= 3 else movies[-2]

        for m, r in zip(movies[:-2], ratings[:-2]):
            train_events.append((u, m, r))

        val_next.append((u, last_train_for_val, val_movie))
        test_next.append((u, last_train_for_test, test_movie))

    train_events = np.array(train_events, dtype=np.int64)
    val_next = np.array(val_next, dtype=np.int64)
    test_next = np.array(test_next, dtype=np.int64)
    return train_events, val_next, test_next


def build_movie_user_matrix(train_events, num_movies, num_users, use_ratings=True):
    mat = np.zeros((num_movies, num_users), dtype=np.float32)
    counts = np.zeros((num_movies, num_users), dtype=np.int32)

    for u, m, r in train_events:
        if use_ratings:
            mat[m, u] += float(r)
        else:
            mat[m, u] += 1.0
        counts[m, u] += 1

    non_zero = counts > 0
    if use_ratings:
        mat[non_zero] /= counts[non_zero]

    return mat
