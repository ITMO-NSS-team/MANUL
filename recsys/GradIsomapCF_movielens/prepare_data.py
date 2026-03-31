from collections import defaultdict

import numpy as np


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
