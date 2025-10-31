import os
import sys
import pandas as pd

# Ensure the app directory (your working directory) is on sys.path
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from main import detect_ai_single_image  # main.py lives inside app/

def detect_ai_in_folder(im_dir):
    results = []
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    image_files = [f for f in os.listdir(im_dir) if f.lower().endswith(valid_exts)]

    if not image_files:
        print("⚠️  No image files found in the specified directory.")
        return

    print(f"=== AI DETECTION FOR {len(image_files)} IMAGES ===")

    for fname in image_files:
        path = os.path.join(im_dir, fname)
        try:
            result = detect_ai_single_image(path)
            if isinstance(result, dict):
                result["filename"] = fname
            else:
                result = {"filename": fname, "result": result}
            results.append(result)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    df = pd.DataFrame(results)
    print("\n=== SUMMARY RESULTS ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    # Point to your images directory
    im_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources"))
    detect_ai_in_folder(im_dir)
