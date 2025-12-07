import os
import time
import numpy as np
import pandas as pd
from PIL import Image

from skimage.filters import sobel, prewitt, gaussian, gabor
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import graycomatrix, graycoprops, hessian_matrix

import multiprocessing as mp
import argparse


def extract_features_one(image_path, label):
    try:
        img = Image.open(image_path).convert("L")
        img = np.array(img)

        features = {}

        # First-order / basic filters
        features["entropy"] = float(np.mean(entropy(img, disk(5))))
        features["gaussian"] = float(np.mean(gaussian(img, sigma=1)))
        features["sobel"] = float(np.mean(sobel(img)))
        features["prewitt"] = float(np.mean(prewitt(img)))

        # Hessian-based feature (using skimage)
        Hxx, Hxy, Hyy = hessian_matrix(img, sigma=1, order='rc', use_gaussian_derivatives=False)
        features["hessian_mean"] = float(np.mean(Hxx + Hyy))

        # Gabor filter (texture feature) — using skimage.gabor
        g_real, g_imag = gabor(img, frequency=0.6)
        features["gabor_mean"] = float(np.mean(g_real))

        # Gray-Level Co-occurrence Matrix (GLCM) features
        glcm = graycomatrix(
            img,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )

        glcm_props = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
        for p in glcm_props:
            val = graycoprops(glcm, p)
            features[f"glcm_{p}"] = float(np.mean(val))

        # Label & ID
        features["label"] = label
        features["image_id"] = os.path.basename(image_path)

        return features

    except Exception as e:
        print("Error:", image_path, e)
        return None


def extract_all(input_dir, output_path):
    start = time.time()

    yes_dir = os.path.join(input_dir, "yes")
    no_dir = os.path.join(input_dir, "no")

    items = []
    if os.path.isdir(yes_dir):
        for f in os.listdir(yes_dir):
            items.append((os.path.join(yes_dir, f), "yes"))
    if os.path.isdir(no_dir):
        for f in os.listdir(no_dir):
            items.append((os.path.join(no_dir, f), "no"))

    print("Total images:", len(items))

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(extract_features_one, items)

    results = [r for r in results if r is not None]

    df = pd.DataFrame(results)
    df.to_parquet(output_path, index=False)

    end = time.time()

    print("num_images:", len(df))
    print("num_features:", df.shape[1] - 2)  # subtract label & image_id
    print("extraction_time_seconds:", round(end - start, 2))
    print("compute_sku:", os.environ.get("AZUREML_COMPUTE", "unknown"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    extract_all(args.input_data, args.output_path)
