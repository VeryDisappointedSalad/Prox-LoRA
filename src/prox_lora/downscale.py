import shutil
from pathlib import Path

import PIL.Image
from tqdm.contrib.concurrent import process_map

"""
changes data/retinopathy/1024 to 224 pixel format
TODO: probably should be moved to utils somewhere
"""

# paths
source_dir = Path("data/retinopathy/1024/")
target_dir = Path("data/retinopathy/224_test/")

# target folders
for p in ["test", "train", "sample"]:
    (target_dir / p).mkdir(parents=True, exist_ok=True)

# copy the .csvs
for p in ["sampleSubmission.csv", "testLabels.csv", "trainLabels.csv"]:
    if (source_dir / p).exists():
        shutil.copyfile(source_dir / p, target_dir / p)

# get all images in the 1024 folder
img_paths = list(source_dir.rglob("*.jpeg"))  # + list(source_dir.rglob("*.JPEG"))


def resize_only(path):
    try:
        rel_path = path.relative_to(source_dir)
        dest_path = target_dir / rel_path

        # just resize the image down
        img = PIL.Image.open(path)
        img = img.resize((224, 224), resample=PIL.Image.Resampling.BILINEAR)
        img.save(dest_path)
    except Exception as e:
        print(f"Błąd przy {path}: {e}")


if __name__ == "__main__":
    # might as well use 12 cores of the CPU
    process_map(resize_only, img_paths, max_workers=12, chunksize=10)
