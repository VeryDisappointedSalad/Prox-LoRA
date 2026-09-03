# type: ignore
import gzip
import json
import shutil
import time
from pathlib import Path

import cv2
import numba
import numpy as np
import PIL.Image
from tqdm.contrib.concurrent import process_map

"""
.py version of preprocess-retinopathy.ipynb
"""

# paths
img_dir = Path("/storage_hdd_1/kc429229/data/retinopathy/original/")
results_path = Path("data/retinopathy/preprocess_results.jsonl.gz")
target_size = 224
target_img_dir = Path(f"data/retinopathy/{target_size}/")

# create folders
Path("data/retinopathy/").mkdir(parents=True, exist_ok=True)
for p in ["test", "train", "sample"]:
    (target_img_dir / p).mkdir(parents=True, exist_ok=True)

# copy .csvs
for p in ["sampleSubmission.csv", "testLabels.csv", "trainLabels.csv"]:
    if (img_dir / p).exists():
        shutil.copyfile(img_dir / p, target_img_dir / p)

# loading images
img_paths = list(img_dir.rglob("*.jpeg"))  # + list(img_dir.rglob("*.JPEG"))
img_paths = sorted(img_paths, key=lambda p: (int(p.stem.split("_")[0]), p.stem))
print(f"found {len(img_paths)} images.")


# circle finding
@numba.njit
def get_circle_scores(img, grad, ys, xs, rs):
    H, W = img.shape
    scores = np.zeros((len(ys), len(xs), len(rs)), dtype=np.float64)
    angles = np.linspace(0, 160, 8)
    directions = np.stack((np.sin(np.deg2rad(angles)), np.cos(np.deg2rad(angles))), axis=-1)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            for ir, r in enumerate(rs):
                scores[iy, ix, ir] = 0
                for d in directions:
                    y0, x0 = int(np.round(y + r * d[0])), int(np.round(x + r * d[1]))
                    y1, x1 = int(np.round(y - r * d[0])), int(np.round(x - r * d[1]))
                    if not (0 <= y0 < H and 0 <= x0 < W and 0 <= y1 < H and 0 <= x1 < W):
                        continue
                    scores[iy, ix, ir] += np.cbrt(
                        np.maximum(grad[y0, x0].dot(-d), 0) * np.maximum(grad[y1, x1].dot(d), 0)
                    )
    return scores


def find_circle(pil_img, ys, xs, rs, downscale):
    pil_img = pil_img.reduce(downscale)
    # H, W = pil_img.height, pil_img.width
    img = np.array(pil_img).max(axis=2)

    grad = np.stack((cv2.Scharr(img, cv2.CV_64F, 0, 1), cv2.Scharr(img, cv2.CV_64F, 1, 0)), axis=-1)
    grad = grad / (img[..., np.newaxis] / img.max() + 0.2)
    K = 25
    grad = np.ascontiguousarray(cv2.GaussianBlur(grad, (K, K), 0))

    scores = get_circle_scores(img, grad, ys / downscale, xs / downscale, rs / downscale)
    b = scores.argmax()
    best_iy, best_ix, best_ir = b // (len(rs) * len(xs)), (b // len(rs)) % len(xs), b % len(rs)
    return float(ys[best_iy]), float(xs[best_ix]), float(rs[best_ir])


def find_circle_iterative(pil_img, k=9, iterations=5):
    H, W = pil_img.height, pil_img.width
    min_y, max_y = int(0.45 * H), int(0.55 * H)
    min_x, max_x = int(0.45 * W), int(0.55 * W)
    min_r, max_r = int(0.4 * min(H, W)), max(H, W) // 2

    best_y, best_x, best_r = (min_y + max_y) / 2, (min_x + max_x) / 2, (min_r + max_r) / 2

    for i in range(iterations):
        ys, y_step = np.linspace(min_y, max_y, k, retstep=True)
        xs, x_step = np.linspace(min_x, max_x, k + 2, retstep=True)
        rs, r_step = np.linspace(min_r, max_r, 2 * k, retstep=True)

        target_size_local = 1536 // 2 ** min(iterations - 1 - i, 2)
        best_y, best_x, best_r = find_circle(pil_img, ys, xs, rs, downscale=max(1, min(H, W) // target_size_local))

        min_y, max_y = best_y - 1.2 * y_step, best_y + 1.2 * y_step
        min_x, max_x = best_x - 1.2 * x_step, best_x + 1.2 * x_step
        min_r, max_r = best_r - 1.2 * r_step, best_r + 1.2 * r_step

    return round(best_y), round(best_x), round(best_r)


# single image parsing
def process(img_path):
    try:
        pil_img = PIL.Image.open(img_path)
        cy, cx, r = find_circle_iterative(pil_img)
        img = np.array(pil_img).max(axis=2)
        y, x = np.indices(img.shape)
        in_mask = ((y - cy) ** 2 + (x - cx) ** 2) <= r**2
        in_p_dark = (img[in_mask] < 30).mean()
        out_p_dark = (img[~in_mask] < 30).mean()

        return {
            "path": str(img_path.relative_to(img_dir)),
            "h": pil_img.height,
            "w": pil_img.width,
            "cy": cy,
            "cx": cx,
            "r": r,
            "in_p_dark": round(float(in_p_dark), 4),
            "out_p_dark": round(float(out_p_dark), 4),
        }
    except Exception as e:
        print(f"Error parsing {img_path}: {e}")
        return None


def crop_and_rescale_and_save(item):
    img_path, res = item
    if res is None:
        return
    try:
        pil_img = PIL.Image.open(img_path)
        cy, cx, r = res["cy"], res["cx"], res["r"]
        H, W = pil_img.height, pil_img.width

        crop_r = int(1.01 * r)
        left, right = max(0, cx - crop_r), min(W, cx + crop_r)
        top, bottom = max(0, cy - crop_r), min(H, cy + crop_r)

        cropH, cropW = bottom - top, right - left
        if cropW >= cropH:
            newH, newW = int(target_size * cropH / cropW), target_size
        else:
            newH, newW = target_size, int(target_size * cropW / cropH)

        pil_img = pil_img.resize((newW, newH), box=(left, top, right, bottom), resample=PIL.Image.Resampling.BILINEAR)
        if newH > newW:
            pil_img = pil_img.rotate(90, expand=True)

        target_img_path = target_img_dir / res["path"]
        pil_img.save(target_img_path)
    except Exception as e:
        print(f"Błąd zapisu/cropowania {img_path}: {e}")


if __name__ == "__main__":
    start_time = time.time()

    print("Looking for circles...")
    results_list = process_map(process, img_paths, max_workers=12, chunksize=4, smoothing=0.01)

    # filtering errors
    results_list = [r for r in results_list if r is not None]
    with gzip.open(results_path, "wb") as f:
        for d in results_list:
            f.write(json.dumps(d).encode() + b"\n")
    print(f"Saved to {results_path}")

    # scaling
    print(f"Cutting images to fit {target_size}...")
    paired_data = list(zip(img_paths, results_list, strict=True))
    process_map(crop_and_rescale_and_save, paired_data, max_workers=12, chunksize=4, smoothing=0.01)

    print(f"Ended in {time.time() - start_time:.2f}s!")
