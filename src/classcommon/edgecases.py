"""Classification pipelines: checks for the cases that silently ruin a small-data evaluation.

* duplicates: the same footage uploaded twice must never sit in a training fold and a test fold (`dup_groups`: 64-bit average hash of one frame + union-find)
* too few clips: `check_enough` says why no metric is reported instead of printing an unstable number
* clips that cannot be trusted as they are: `flag_records` (very short, no pair, no result) so the report can show metrics with and without them
* label words in a document: `leak_scan`
"""
import re

import cv2
import numpy as np

SHORT_S = 1.5


def ahash(video_path, size=8):
    """64-bit average hash (as a Python int) of the middle frame, or None when the video cannot be read."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, img = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA)
    bits = (g > g.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def dup_groups(hashes, max_dist=3):
    """hashes: {clip_id: int or None} -> {clip_id: group id (0..)}. Clips whose hashes differ in at most max_dist bits share a group (transitively)."""
    ids = list(hashes)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    have = [i for i in ids if hashes[i] is not None]
    for x in range(len(have)):
        for y in range(x + 1, len(have)):
            if hamming(hashes[have[x]], hashes[have[y]]) <= max_dist:
                parent[find(have[x])] = find(have[y])
    roots = {}
    return {i: roots.setdefault(find(i), len(roots)) for i in ids}


def check_enough(y, min_per_class):
    """-> (ok, message). A class with fewer than min_per_class clips cannot be cross-validated honestly."""
    y = np.asarray(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if min(n1, n0) < min_per_class:
        return False, (f"not evaluated: {n1} violent and {n0} non-violent clips; at least {min_per_class} of each are needed for a cross-validated result "
                       f"(add start times / clips and run again)")
    return True, f"{n1} violent and {n0} non-violent clips"


def flag_records(recs):
    """{clip_id: [flags]}: 'short' (under SHORT_S s), 'no_pair' (no interacting pair measured), 'no_result' (no analysis result at all)."""
    out = {}
    for r in recs:
        f = []
        if r["duration_s"] < SHORT_S:
            f.append("short")
        if r.get("result") is None:
            f.append("no_result")
        elif r["result"].get("no_pair"):
            f.append("no_pair")
        out[r["clip_id"]] = f
    return out


def leak_scan(docs, pattern):
    """{clip_id: [matched words]} for the documents that contain a word of the compiled regex `pattern`."""
    rx = pattern if hasattr(pattern, "findall") else re.compile(pattern, re.I)
    return {k: sorted({m.lower() if isinstance(m, str) else m[0].lower() for m in rx.findall(v)}) for k, v in docs.items() if rx.search(v)}
