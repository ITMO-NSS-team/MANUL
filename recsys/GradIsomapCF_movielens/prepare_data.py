from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd


def _sort_order(user_idx, timestamp):
    return np.lexsort((np.asarray(timestamp), np.asarray(user_idx)))


def _build_user2seq(u, m, ts, r):
    user2seq = defaultdict(list)
    if u.size == 0:
        return user2seq
    cuts = np.flatnonzero(np.diff(u)) + 1
    starts = np.concatenate(([0], cuts)).tolist()
    ends = np.concatenate((cuts, [u.size])).tolist()
    triples = list(zip(m.tolist(), ts.tolist(), r.tolist()))
    u_first = u[starts].tolist()
    for uid, s, e in zip(u_first, starts, ends):
        user2seq[int(uid)] = triples[s:e]
    return user2seq


def prepare_sequences(df, build_user2seq=True, dedup=True):
    user_codes, _ = pd.factorize(df["userId"], sort=True)
    movie_codes, _ = pd.factorize(df["movieId"], sort=True)
    df = df.assign(user_idx=user_codes.astype(np.int64),
                   movie_idx=movie_codes.astype(np.int64))
    order = _sort_order(df["user_idx"].to_numpy(), df["timestamp"].to_numpy())
    df = df.iloc[order]

    if dedup:
        dup = df.duplicated(["user_idx", "movie_idx"], keep="first").to_numpy()
        n_dup = int(dup.sum())
        if n_dup:
            print(f"  [prepare_sequences] удалено {n_dup} повторных пар (user, item) "
                  f"из {len(df)} (оставлено первое по времени взаимодействие)")
            df = df.iloc[np.flatnonzero(~dup)]

    if not build_user2seq:
        return df, defaultdict(list)

    user2seq = _build_user2seq(
        df["user_idx"].to_numpy(), df["movie_idx"].to_numpy(),
        df["timestamp"].to_numpy(), df["rating"].to_numpy(),
    )
    return df, user2seq


def _select_ids(counts, min_count, k, legacy_tie_break=False):
    ids = np.flatnonzero((counts >= min_count) & (counts > 0))
    if ids.size > k:
        if legacy_tie_break:
            s = pd.Series(counts[ids], index=ids)
            ids = np.sort(s.sort_values(ascending=False).head(k).index.values)
        else:
            ids = ids[np.argsort(-counts[ids], kind="stable")[:k]]
    return ids


def subsample_users_items(df_mapped,
                          max_users=500,
                          max_movies=1000,
                          min_seq_len=5,
                          legacy_tie_break=False):
    u = df_mapped["user_idx"].to_numpy()
    m = df_mapped["movie_idx"].to_numpy()
    n_u = int(u.max()) + 1 if u.size else 0
    n_m = int(m.max()) + 1 if m.size else 0

    good_users = _select_ids(np.bincount(u, minlength=n_u), min_seq_len, max_users, legacy_tie_break)
    keep_u = np.zeros(n_u, dtype=bool)
    keep_u[good_users] = True
    mask1 = keep_u[u]

    m1 = m[mask1]
    good_movies = _select_ids(np.bincount(m1, minlength=n_m), 1, max_movies, legacy_tie_break)
    keep_m = np.zeros(n_m, dtype=bool)
    keep_m[good_movies] = True
    mask2 = keep_m[m1]

    u2 = u[mask1][mask2]
    good_users2 = _select_ids(np.bincount(u2, minlength=n_u), min_seq_len, u2.size + 1)
    keep_u2 = np.zeros(n_u, dtype=bool)
    keep_u2[good_users2] = True
    mask3 = keep_u2[u2]

    rows = np.flatnonzero(mask1)[mask2][mask3]          # позиции в df_mapped
    df_sub = df_mapped.iloc[rows].copy()

    u3 = u2[mask3]
    m3 = m1[mask2][mask3]
    old_user_ids, new_u = np.unique(u3, return_inverse=True)
    old_movie_ids, new_m = np.unique(m3, return_inverse=True)
    df_sub["user_idx"] = new_u.astype(np.int64)
    df_sub["movie_idx"] = new_m.astype(np.int64)

    num_users = int(old_user_ids.size)
    num_movies = int(old_movie_ids.size)

    order = _sort_order(df_sub["user_idx"].to_numpy(), df_sub["timestamp"].to_numpy())
    df_sub = df_sub.iloc[order]

    user2seq_sub = _build_user2seq(
        df_sub["user_idx"].to_numpy(), df_sub["movie_idx"].to_numpy(),
        df_sub["timestamp"].to_numpy(), df_sub["rating"].to_numpy(),
    )

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


def _events_dtype(ratings):
    r = np.asarray(ratings)
    if r.size == 0 or np.issubdtype(r.dtype, np.integer):
        return np.int64
    return np.int64 if bool(np.all(r == np.floor(r))) else np.float64


def _split_arrays(u, m, r, min_len):
    n = u.size
    empty3 = lambda dt: np.empty((0, 3), dtype=dt)
    dt = _events_dtype(r)
    if n == 0:
        return empty3(dt), empty3(dt), empty3(dt)

    cuts = np.flatnonzero(np.diff(u)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [n]))
    lens = ends - starts
    ok = lens >= min_len

    s, e, l = starts[ok], ends[ok], lens[ok]
    last = e - 1
    users = u[s]

    test = np.stack([users, m[last - 1], m[last]], axis=1)
    val_prev = np.where(l >= 3, last - 2, last - 1)
    val = np.stack([users, m[val_prev], m[last - 1]], axis=1)

    rep_start = np.repeat(starts, lens)
    rep_len = np.repeat(lens, lens)
    pos = np.arange(n) - rep_start
    tr_mask = (rep_len >= min_len) & (pos < rep_len - 2)
    train = np.stack([u[tr_mask], m[tr_mask], r[tr_mask]], axis=1)

    return train.astype(dt), val.astype(np.int64), test.astype(np.int64)


def train_val_test_split_next_item(user2seq, min_len=3):
    """user2seq: {user: [(movie, ts, rating), ...]} (в порядке времени)."""
    users = list(user2seq.keys())
    lens = np.fromiter((len(user2seq[k]) for k in users), dtype=np.int64, count=len(users))
    if lens.sum() == 0:
        return _split_arrays(np.empty(0, np.int64), np.empty(0, np.int64),
                             np.empty(0, np.int64), min_len)
    flat = np.array(list(chain.from_iterable(user2seq[k] for k in users)))
    u = np.repeat(np.asarray(users, dtype=np.int64), lens)
    m = flat[:, 0].astype(np.int64)
    r = flat[:, 2]
    if np.all(r == np.floor(r)):
        r = r.astype(np.int64)
    return _split_arrays(u, m, r, min_len)


def train_val_test_split_next_item_df(df_sub, min_len=3):
    return _split_arrays(
        df_sub["user_idx"].to_numpy(dtype=np.int64),
        df_sub["movie_idx"].to_numpy(dtype=np.int64),
        df_sub["rating"].to_numpy(),
        min_len,
    )


def build_movie_user_matrix(train_events, num_movies, num_users, use_ratings=True):
    ev = np.asarray(train_events)
    if ev.size == 0:
        return np.zeros((num_movies, num_users), dtype=np.float32)
    u = ev[:, 0].astype(np.int64)
    m = ev[:, 1].astype(np.int64)
    r = ev[:, 2].astype(np.float64)

    flat = m * num_users + u
    size = num_movies * num_users
    counts = np.bincount(flat, minlength=size)
    if use_ratings:
        sums = np.bincount(flat, weights=r, minlength=size)
        mat = np.zeros(size, dtype=np.float64)
        nz = counts > 0
        mat[nz] = sums[nz] / counts[nz]
    else:
        mat = counts.astype(np.float64)
    return mat.reshape(num_movies, num_users).astype(np.float32)


def thin_train_events(train_events, keep_frac, seed=0, min_keep=1):
    ev = np.asarray(train_events)
    if keep_frac >= 1.0 or ev.shape[0] == 0:
        return ev
    if not 0.0 < keep_frac < 1.0:
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
    u = ev[:, 0].astype(np.int64)
    r = np.random.default_rng(seed).random(ev.shape[0])
    order = np.lexsort((r, u))                       # по пользователю, внутри - случайно
    su = u[order]
    first = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    counts = np.diff(np.r_[first, su.size])
    k = np.floor(keep_frac * counts + 0.5).astype(np.int64)
    k = np.minimum(np.maximum(k, min_keep), counts)
    pos = np.arange(su.size) - np.repeat(first, counts)
    keep_sorted = pos < np.repeat(k, counts)
    mask = np.zeros(ev.shape[0], dtype=bool)
    mask[order[keep_sorted]] = True
    return ev[mask]


def corater_stats(pairs, num_items, num_users, max_items=8000):
    if num_items > max_items:
        return None
    p = np.asarray(pairs)[:, :2].astype(np.int64)
    B = np.zeros((num_items, num_users), dtype=np.float32)
    B[p[:, 1], p[:, 0]] = 1.0
    C = B @ B.T
    iu = np.triu_indices(num_items, k=1)
    c = C[iu]
    deg = B.sum(axis=1)
    return dict(
        pairs=int(c.size),
        corater_median=float(np.median(c)),
        corater_mean=float(c.mean()),
        frac_pairs_zero=float((c == 0).mean()),
        frac_pairs_le3=float((c <= 3).mean()),
        item_raters_median=float(np.median(deg)),
        frac_items_lt10_raters=float((deg < 10).mean()),
    )


def popularity_baseline(items, labels, train_pairs, num_items, top_k=10):
    items = np.asarray(items)
    labels = np.asarray(labels)
    pairs = np.asarray(train_pairs)[:, :2].astype(np.int64)
    pop = np.bincount(pairs[:, 1], minlength=num_items).astype(np.float64)
    sc = pop[items]
    neg_valid = labels[:, 1:] >= 0
    pos = sc[:, :1]
    greater = ((sc[:, 1:] > pos) & neg_valid).sum(axis=1)
    ties = ((sc[:, 1:] == pos) & neg_valid).sum(axis=1)
    k = top_k
    m = 1 + neg_valid.sum(axis=1)
    return dict(
        top_k=int(k), n_users=int(items.shape[0]),
        random_hr=float(np.mean(np.minimum(k, m) / m)),
        popularity_hr=float(np.mean(np.clip((k - greater) / (ties + 1.0), 0.0, 1.0))),
        popularity_hr_pessimistic=float(np.mean(greater + ties < k)),
        popularity_hr_optimistic=float(np.mean(greater < k)),
    )
