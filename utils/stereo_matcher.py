from utils.data_loader import shared_track_ratio


def best_pairs(imgs, pair_B, k=5, min_B=0.15, max_B=0.6, min_overlap=0.7):
    """Find top-k pairs with good baseline and overlap"""
    pairs = []
    for (i, j), B in pair_B.items():
        if not (min_B <= B <= max_B):
            continue
        ov = shared_track_ratio(imgs[i], imgs[j])
        if ov < min_overlap:
            continue
        score = B * ov
        pairs.append(((i, j), score))

    # Sort by score and return top-k pairs
    pairs.sort(key=lambda x: x[1], reverse=True)
    if not pairs:
        return []
    print(f"Score of {pairs[:k][0][0]}: {pairs[:k][0][1]}")

    return [p[0] for p in pairs[:k]]


def find_neighbor_views(imgs, pair_B, ref_id, k=3, min_B=0.1, max_B=0.4, min_overlap=0.5):
    """Find k neighboring views for multi-view fusion"""
    neighbors = []
    for (i, j), B in pair_B.items():
        if i == ref_id and min_B <= B <= max_B:
            ov = shared_track_ratio(imgs[i], imgs[j])
            if ov >= min_overlap:
                neighbors.append((j, B * ov))
        elif j == ref_id and min_B <= B <= max_B:
            ov = shared_track_ratio(imgs[i], imgs[j])
            if ov >= min_overlap:
                neighbors.append((i, B * ov))

    neighbors.sort(key=lambda x: x[1], reverse=True)
    return [n[0] for n in neighbors[:k]]
