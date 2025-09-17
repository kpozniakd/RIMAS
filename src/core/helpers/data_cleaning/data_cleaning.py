import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

def estimate_stroke_thickness(binary_img):
    if binary_img.mean() > 127:
        binary_img = cv2.bitwise_not(binary_img)
    binary = (binary_img > 0).astype(np.uint8)
    skeleton = skeletonize(binary)
    area_original = np.sum(binary)
    area_skeleton = np.sum(skeleton)
    if area_skeleton == 0:
        return 0
    return area_original / area_skeleton

def load_and_process_images_from_dir(directory, exts=(".png", ".jpg", ".jpeg")):
    filenames, images, thicknesses = [], [], []

    for fname in tqdm(os.listdir(directory), desc="Check images"):
        if not fname.lower().endswith(exts):
            continue
        path = os.path.join(directory, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thickness = estimate_stroke_thickness(binary)
        thicknesses.append(thickness)
        images.append(img)
        filenames.append(fname)

    return filenames, images, thicknesses


def remove_top_black_rows(img):
    while img.shape[0] > 0 and not np.any(img[0] == 255):
        img = img[1:]
    return img

def estimate_cleaned_thickness(img):
    cleaned = remove_top_black_rows(img)
    _, binary = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return estimate_stroke_thickness(binary), cleaned

def clean_data(df: pd.DataFrame, img_dir: str):
    filenames, images, thicknesses = load_and_process_images_from_dir(img_dir)
    df_lines = pd.DataFrame({
        "filename": filenames,
        "image": images,
        "thickness": thicknesses
    })
    thicknesses_cleaned = []
    images_cleaned = []

    for img in df_lines["image"]:
        thickness, cleaned_img = estimate_cleaned_thickness(img)
        thicknesses_cleaned.append(thickness)
        images_cleaned.append(cleaned_img)

    df_lines_cleaned = pd.DataFrame({
        "filename": df_lines["filename"],
        "image": images_cleaned,
        "thickness": thicknesses_cleaned
    })
    Q1 = df_lines_cleaned["thickness"].quantile(0.25)
    Q3 = df_lines_cleaned["thickness"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (df_lines_cleaned["thickness"] < lower_bound) | (df_lines_cleaned["thickness"] > upper_bound)

    num_outliers = outliers.sum()
    total = len(df_lines_cleaned)
    percentage_outliers = (num_outliers / total) * 100

    print(f"Number of outliers: {num_outliers} з {total}")
    print(f"% of outliers: {percentage_outliers:.2f}%")
    df_merged = pd.merge(df, df_lines_cleaned,
                     left_on="Filenames",
                     right_on="filename",
                     how="inner")

    df_merged.drop(columns=["filename"], inplace=True)

    return df_merged

def build_image_label_dataset(df: pd.DataFrame, img_dir: str) -> pd.DataFrame:
    """
    Зчитує зображення з диска за ім'ям у колонці 'Filenames',
    бінаризує, і повертає DataFrame з 'image' та 'label'.
    """
    images = []
    labels = []
    filenames = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        path = os.path.join(img_dir, row["Filenames"])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        images.append(binary)
        labels.append(row["Contents"])
        filenames.append(row["Filenames"])

    return pd.DataFrame({
        "image": images,
        "label": labels, 
        "filename": filenames
    })
