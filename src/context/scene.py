"""Deep context, Stage B: what is the SCENE? (segmentation layout + monocular depth)  -  report Phase II ideas used early, without prediction.

Layout (SegFormer-B0 trained on ADE20K, 150 classes): a few keyframes per clip are segmented and merged by majority vote into one label map in
normalised image coordinates with these groups:
    walkable  floor, road, sidewalk, path, grass ...      obstacle  wall, building, fence, railing, cabinet ...
    door      door, screen door                           vehicle   car, bus, truck, bicycle ...
    sky, vegetation, furniture (indoor cue), other
From it: place type (indoor / outdoor / unknown) with a confidence, where the doors are (real "exits" instead of "nearest frame edge"), and what is an obstacle.
Dark or unusual scenes are segmented badly (a night alley came out as 'wall 68%, floor 31%'), so every layout carries `layout_conf` (mean top-class
probability) and `reliable` (False for low confidence or a moving camera); downstream code must treat unreliable layouts as unknown.

Depth (Depth Anything V2 Small, the report's Algorithm 2 / DAIR component) is run on a few frames of the key pair only:
    depth of each person (median relative depth on the lower torso), which person is nearer, whether the two are on the same depth plane
    (a pair that overlaps in the image but sits at different depths is NOT close), and whether a person is pinned against an obstacle
    (an obstacle right behind them in the image AND at the same depth).
Relative depth has an unknown scale per frame, so only orderings and small differences are used, never absolute meters.

Outputs per clip (context_dir/<Category>/):  <clip>_layout.json, <clip>_layout.npz (128x128 label map),  <clip>_depth.json (key pair)
Usage:  python -m src.context.scene [--config CFG] [--clips A B] [--force] [--no-depth] [--device cpu|cuda]
"""
import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.assm.track_poses import load_tracks
from src.config import load_config, resolve_path
from src.context.features import load_context, rank_pairs
from src.video.shots import SHOT_BASE

log = logging.getLogger(__name__)

GROUP = {"other": 0, "walkable": 1, "obstacle": 2, "door": 3, "vehicle": 4, "sky": 5, "vegetation": 6, "furniture": 7}
NAME_GROUPS = {
    "walkable": {"floor", "flooring", "road", "route", "sidewalk", "pavement", "path", "grass", "field", "earth", "ground", "sand", "rug", "carpet", "runway", "stage", "dirt track", "land"},
    "obstacle": {"wall", "building", "edifice", "fence", "railing", "column", "pillar", "bookcase", "cabinet", "counter", "shelf", "wardrobe", "house", "skyscraper", "tower", "bannister", "booth", "kitchen island"},
    "door": {"door", "screen door"},
    "vehicle": {"car", "bus", "truck", "van", "minibike", "bicycle", "boat", "airplane", "ship"},
    "sky": {"sky"},
    "vegetation": {"tree", "plant", "flora", "palm", "flower"},
    "furniture": {"ceiling", "bed", "sofa", "couch", "chair", "armchair", "table", "desk", "curtain", "cushion", "lamp", "television", "pool table", "bathtub", "stove", "refrigerator"},
}
ROAD_LIKE = {"road", "route", "sidewalk", "pavement", "path"}


def group_lut(id2label):
    """numpy array mapping class id -> group code, from the model's own class names (ADE20K names can be 'a, b' synonyms)."""
    lut = np.zeros(max(int(k) for k in id2label) + 1, np.uint8)
    for k, name in id2label.items():
        words = {w.strip() for w in str(name).split(",")}
        for g, names in NAME_GROUPS.items():
            if words & names:
                lut[int(k)] = GROUP[g]
                break
    return lut


def road_like_ids(id2label):
    return {int(k) for k, name in id2label.items() if {w.strip() for w in str(name).split(",")} & ROAD_LIKE}


# ---------------------------------------------------------------- layout facts (pure functions on label maps)
def light_facts(grays):
    """Robust light statistics from grayscale keyframes: median luma and the share of very dark pixels.

    The MEAN is fooled by a single street lamp (a night scene read as daylight), so low light = dark median or mostly dark pixels.
    """
    if not grays:
        return {"median_luma": None, "dark_frac": None, "low_light": None}
    med = float(np.median([np.median(g) for g in grays]))
    dark = float(np.mean([(g < 50).mean() for g in grays]))
    return {"median_luma": round(med, 1), "dark_frac": round(dark, 3), "low_light": bool(med < 70 or dark > 0.55)}


def layout_facts(labels, conf, road_frac=0.0, moving_camera=False, min_conf=0.5, light=None):
    """Facts from a merged group-label map (H,W uint8, normalised coordinates)."""
    n = labels.size
    frac = {g: float((labels == c).sum() / n) for g, c in GROUP.items()}
    sky, veg = frac["sky"], frac["vegetation"]
    outdoor = sky > 0.03 or road_frac > 0.15 or veg > 0.08
    indoor = frac["furniture"] > 0.08
    place = "outdoor" if outdoor and not indoor else "indoor" if indoor and not outdoor else "unknown"
    doors = []
    mask = (labels == GROUP["door"]).astype(np.uint8)
    cnt, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = labels.shape
    for k in range(1, cnt):
        x, y, w, h, area = stats[k]
        if area / n >= 0.004 and h >= 0.6 * w:        # door-shaped: at least as tall as it is wide (roughly)
            doors.append({"x1": x / W, "y1": y / H, "x2": (x + w) / W, "y2": (y + h) / H, "cx": (x + w / 2) / W, "base_y": (y + h) / H,
                          "area": float(area / n)})
    doors.sort(key=lambda d: -d["area"])
    return {"fractions": {k: round(v, 3) for k, v in frac.items()}, "road_like_frac": round(float(road_frac), 3), "place_type": place,
            "doors": doors[:4], "walkable_frac": round(frac["walkable"], 3), "layout_conf": round(float(conf), 3), "light": light or light_facts([]),
            "reliable": bool(conf >= min_conf and not moving_camera)}


def group_at(labels, x_norm, y_norm):
    """Group code at a normalised image point (clamped); 0 outside."""
    H, W = labels.shape
    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        return 0
    return int(labels[min(H - 1, int(y_norm * H)), min(W - 1, int(x_norm * W))])


def blocks_door(foot_i, foot_j, door_base, h_px, corridor_h=0.35):
    """Is person j standing between person i and a door? All points in pixels (feet / door base); corridor = share of body height."""
    a, b, c = np.asarray(foot_i, float), np.asarray(door_base, float), np.asarray(foot_j, float)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-6:
        return False
    s = float((c - a) @ ab) / L2
    if not 0.05 < s < 0.95:
        return False
    return bool(np.linalg.norm(c - (a + s * ab)) <= corridor_h * h_px)


# ---------------------------------------------------------------- depth helpers (pure functions on a depth map)
def person_depth(dm, bbox):
    """Median relative depth (0-1, larger = nearer) on the lower torso of a person box. dm: (H,W) float array in video pixels."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    xs, xe = int(max(0, cx - 0.15 * w)), int(min(dm.shape[1], cx + 0.15 * w) + 1)
    ys, ye = int(max(0, y1 + 0.45 * h)), int(min(dm.shape[0], y1 + 0.75 * h) + 1)
    patch = dm[ys:ye, xs:xe]
    return float(np.median(patch)) if patch.size else np.nan


def pinned_against(dm, labels, bbox, other_cx, tol=0.10):
    """Is this person pinned against an obstacle on the side AWAY from the other person?

    Samples the layout a little beyond the person's box on the far side; it counts only if the layout says obstacle / vehicle there AND the depth
    there matches the person's depth (a distant background wall that merely appears behind them does not count).
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    side = -1.0 if other_cx > cx else 1.0
    px = cx + side * (0.5 * w + 0.2 * h)
    py = y1 + 0.5 * h
    H, W = dm.shape
    if not (0 <= px < W and 0 <= py < H):
        return False
    g = group_at(labels, px / W, py / H)
    if g not in (GROUP["obstacle"], GROUP["vehicle"]):
        return False
    d_pt = float(np.median(dm[max(0, int(py) - 3):int(py) + 4, max(0, int(px) - 3):int(px) + 4]))
    d_me = person_depth(dm, bbox)
    return bool(np.isfinite(d_me) and abs(d_pt - d_me) <= tol)


# ---------------------------------------------------------------- models (lazy, optional)
class SceneModels:
    """Loads SegFormer and Depth Anything on first use. Needs the `transformers` package and downloads weights once."""

    SEG = "nvidia/segformer-b0-finetuned-ade-512-512"
    DEPTH = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, device="auto"):
        import torch
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self._seg = self._proc = self._depth = self._lut = self._road_ids = None

    def _load_seg(self):
        if self._seg is None:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            self._proc = SegformerImageProcessor.from_pretrained(self.SEG)
            self._seg = SegformerForSemanticSegmentation.from_pretrained(self.SEG).eval().to(self.device)
            self._lut = group_lut(self._seg.config.id2label)
            self._road_ids = road_like_ids(self._seg.config.id2label)

    def segment(self, frame_bgr):
        """-> (group map (128,128) uint8, mean top-class probability, share of road-like pixels)."""
        self._load_seg()
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        with self.torch.no_grad():
            inp = self._proc(images=rgb, return_tensors="pt").to(self.device)
            logits = self._seg(**inp).logits
            prob = logits.softmax(1)
            conf, cls = prob.max(1)
        cls = cls[0].cpu().numpy()
        road = float(np.isin(cls, list(self._road_ids)).mean())
        return self._lut[cls], float(conf.mean()), road

    def depth(self, frame_bgr):
        """-> relative depth (H,W) float32 in [0,1], larger = nearer, at the frame's own resolution."""
        if self._depth is None:
            from transformers import pipeline
            self._depth = pipeline("depth-estimation", model=self.DEPTH, device=0 if self.device == "cuda" else -1)
        from PIL import Image
        d = np.array(self._depth(Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1])))["depth"], np.float32)
        d = (d - d.min()) / max(float(d.max() - d.min()), 1e-6)
        return cv2.resize(d, (frame_bgr.shape[1], frame_bgr.shape[0]))


# ---------------------------------------------------------------- per clip
def read_frames(video_path, frame_ids):
    cap = cv2.VideoCapture(str(video_path))
    out = {}
    for f in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            out[int(f)] = img
    cap.release()
    return out


def key_shot_range(t, scene, key_pair):
    """(start_frame, end_frame) of the shot to analyse: the one containing the key pair, else the longest shot."""
    n = scene["n_frames"]
    starts = [int(x) for x in t["shot_starts"]] if "shot_starts" in t else [0]
    ends = [x - 1 for x in starts[1:]] + [n - 1]
    if key_pair:
        shot = int(key_pair[0]) // SHOT_BASE
        if shot < len(starts):
            return starts[shot], ends[shot], shot
    k = int(np.argmax([e - s for s, e in zip(starts, ends)]))
    return starts[k], ends[k], k


def clip_layout(video_path, frame_range, models, camera_moving, n_keyframes=5):
    """Majority-vote layout over evenly spaced keyframes of ONE shot (frame_range = (first, last)) -> (labels (128,128), facts dict).

    Keyframes are never taken from different shots: a layout merged across a cut would blend unrelated scenes.
    """
    a, b = frame_range
    ids = np.linspace(a, max(a, b), n_keyframes + 2)[1:-1].astype(int)
    maps, confs, roads, grays = [], [], [], []
    for img in read_frames(video_path, ids).values():
        grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        m, c, r = models.segment(img)
        maps.append(m)
        confs.append(c)
        roads.append(r)
    if not maps:
        return np.zeros((128, 128), np.uint8), {"error": "no frames", "reliable": False, "layout_conf": 0.0, "place_type": "unknown", "doors": []}
    stack = np.stack(maps)
    labels = np.zeros(stack.shape[1:], np.uint8)
    counts = np.stack([(stack == c).sum(0) for c in range(len(GROUP))])
    labels = counts.argmax(0).astype(np.uint8)
    return labels, layout_facts(labels, float(np.mean(confs)), float(np.mean(roads)), camera_moving, light=light_facts(grays))


def pair_depth_samples(t, scene, pair, labels, video_path, models, n_samples, end_frame):
    """Depth cross-check for one pair over a few frames before `end_frame`."""
    i, j = pair
    fi = {int(f): k for k, f in zip(np.where(t["track_id"] == i)[0], t["frame_idx"][t["track_id"] == i])}
    fj = {int(f): k for k, f in zip(np.where(t["track_id"] == j)[0], t["frame_idx"][t["track_id"] == j])}
    common = sorted(set(fi) & set(fj))
    common = [f for f in common if f < end_frame]
    if not common:
        return {"pair": [i, j], "frames": [], "depth_agree_frac": None}
    pick = [common[int(k)] for k in np.linspace(0, len(common) - 1, min(n_samples, len(common)))]
    frames = read_frames(video_path, pick)
    rows = []
    for f in pick:
        if f not in frames:
            continue
        dm = models.depth(frames[f])
        bi, bj = t["bbox"][fi[f]], t["bbox"][fj[f]]
        di, dj = person_depth(dm, bi), person_depth(dm, bj)
        hi, hj = float(bi[3] - bi[1]), float(bj[3] - bj[1])
        rows.append({"t": round(f / scene["fps"], 2), "frame": f, "depth_i": round(di, 3), "depth_j": round(dj, 3),
                     "depth_gap": round(abs(di - dj), 3), "height_gap": round(abs(hi - hj) / max(hi, hj, 1e-6), 3), "nearer_by_depth": "i" if di > dj else "j", "nearer_by_height": "i" if hi > hj else "j",
                     "pinned_i": pinned_against(dm, labels, bi, (bj[0] + bj[2]) / 2), "pinned_j": pinned_against(dm, labels, bj, (bi[0] + bi[2]) / 2)})
    # depth and box height can only disagree meaningfully when BOTH say clearly who is nearer (people side by side tie on depth)
    decisive = [r for r in rows if r["depth_gap"] >= 0.08 and r["height_gap"] >= 0.08]
    agree = float(np.mean([r["nearer_by_depth"] == r["nearer_by_height"] for r in decisive])) if decisive else None
    return {"pair": [i, j], "frames": rows, "depth_agree_frac": agree, "n_decisive": len(decisive)}


def save_layout(out_dir, category, clip_id, labels, facts):
    d = Path(out_dir) / category
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{clip_id}_layout.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    np.savez_compressed(d / f"{clip_id}_layout.npz", labels=labels)


def load_layout(out_dir, category, clip_id):
    """-> (labels (128,128) or None, facts dict or None)."""
    d = Path(out_dir) / category
    jp, npz = d / f"{clip_id}_layout.json", d / f"{clip_id}_layout.npz"
    if not jp.exists() or not npz.exists():
        return None, None
    with np.load(npz) as z:
        return z["labels"], json.loads(jp.read_text(encoding="utf-8"))


def load_depth(out_dir, category, clip_id):
    p = Path(out_dir) / category / f"{clip_id}_depth.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Stage B: scene layout (SegFormer) and key-pair depth (Depth Anything V2 Small)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ctx = cfg["context"]
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    out_dir = resolve_path(ctx["context_dir"])
    models = SceneModels(args.device)
    n_done = n_skip = 0
    place_counts = {}
    for k, c in enumerate(clean["clips"], 1):
        if args.clips and c["clip_id"] not in args.clips:
            continue
        sp = out_dir / c["category"] / f"{c['clip_id']}_scene.json"
        tp = resolve_path(cfg["assm"]["tracks_dir"]) / c["category"] / f"{c['clip_id']}.npz"
        if not sp.exists() or not tp.exists():
            n_skip += 1
            continue
        lp = out_dir / c["category"] / f"{c['clip_id']}_layout.json"
        dp = out_dir / c["category"] / f"{c['clip_id']}_depth.json"
        want_layout = args.force or not lp.exists()
        want_depth = (not args.no_depth) and (args.force or not dp.exists())
        if not (want_layout or want_depth):
            continue
        try:
            scene, _ = load_context(out_dir, c["category"], c["clip_id"])
            video = resolve_path(c["path"])
            t = load_tracks(tp)
            top = rank_pairs(scene, 1)
            key = tuple(int(x) for x in top[0].split("_")) if top else None
            first, last, shot = key_shot_range(t, scene, key)
            if want_layout:
                labels, facts = clip_layout(video, (first, last), models, scene["camera"]["moving"], int(ctx["layout_keyframes"]))
                facts["shot"] = {"index": int(shot), "start_f": int(first), "end_f": int(last)}
                save_layout(out_dir, c["category"], c["clip_id"], labels, facts)
            else:
                labels, facts = load_layout(out_dir, c["category"], c["clip_id"])
            place_counts[facts["place_type"]] = place_counts.get(facts["place_type"], 0) + 1
            if want_depth:
                if top:
                    i, j = key
                    res = pair_depth_samples(t, scene, (i, j), labels, video, models, int(ctx["depth_frames"]), scene["n_frames"])
                else:
                    res = {"pair": None, "frames": [], "depth_agree_frac": None}
                dp.write_text(json.dumps(res, indent=1), encoding="utf-8")
            n_done += 1
        except Exception as e:  # one bad clip must not stop a long GPU run
            log.warning("Skipping %s: %s", c["clip_id"], e)
        if k % 50 == 0:
            print(f"  scene: {k}/{len(clean['clips'])} clips", flush=True)
    print(f"\nScene layout/depth done for {n_done} clips ({n_skip} without context results or tracks). Place types: {place_counts}\n-> {out_dir}")


if __name__ == "__main__":
    main()
