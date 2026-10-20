import sys
sys.path.insert(0, r"c:\Users\Julia\Documents\NSS_lab\MANUL\recsys\GradIsomapCF_movielens")
from new_datasets import NCFTestDatasetSampled

# Pathological case: user 0 has positive-set covering 799/800 items (only 1
# item free), user 1 is normal (few positives). num_ng=99 - user 0 cannot
# possibly get 99 negatives.
num_items = 800
user_pos_all_set = {
    0: set(range(799)),  # items 0..798 all positive, item 799 free
    1: {5, 10, 15},
}
next_triples = [(0, 100, 799), (1, 5, 50)]

ds = NCFTestDatasetSampled(next_triples, num_items, user_pos_all_set, num_ng=99, seed=1)
print("dataset length:", len(ds))
user0_items = [ds.items[i] for i in range(len(ds)) if ds.users[i] == 0]
user1_items = [ds.items[i] for i in range(len(ds)) if ds.users[i] == 1]
print("user0 sampled items (target=799, expect target + at most 1 negative):", user0_items)
print("user1 sampled items count (expect 1 target + 99 negs = 100):", len(user1_items))
assert len(user0_items) <= 2, "user 0 should get at most 1 negative (only 1 free item)"
assert len(user1_items) == 100, "user 1 (normal density) should get full 99 negatives"
print("PASSED: no hang, dense user truncated gracefully, normal user unaffected")
