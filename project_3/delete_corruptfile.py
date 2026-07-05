import os
from PIL import Image
DATA_DIR = "/content/drive/My Drive/PetImages/"
def find_and_remove_corrupt_images(data_dir, delete=True):
    bad_files = []
    total_checked = 0
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                continue
            fpath = os.path.join(root, fname)
            total_checked += 1
            is_bad = False
            try:
                with Image.open(fpath) as img:
                    img.verify()
            except Exception:
                is_bad = True
            if not is_bad:
                try:
                    with Image.open(fpath) as img:
                        img.load()
                except Exception:
                    is_bad = True
            if is_bad:
                bad_files.append(fpath)
    print(f"Checked {total_checked} files, found {len(bad_files)} corrupt.")
    if delete:
        for f in bad_files:
            try:
                os.remove(f)
                print("Removed:", f)
            except Exception as e:
                print(f"Could not remove {f}: {e}")
    else:
        for f in bad_files:
            print("Would remove:", f)
    return bad_files
bad = find_and_remove_corrupt_images(DATA_DIR, delete=False)
