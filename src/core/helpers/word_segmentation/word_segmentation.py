import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd 
import os

def dilate_image(bw: np.ndarray, kernel_size=(29, 3)) -> np.ndarray:
    """
    Applies morphological dilation to a binarized image to merge close letters into words.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.dilate(bw, kernel, iterations=1)


def load_and_binarize(img):
    """
    Loads an image in grayscale and applies inverted binary thresholding
    (black text on white background becomes white text on black).
    """
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img, bw


def process_image_row(train_set,row_idx=0):

    img, bw = load_and_binarize(train_set.loc[row_idx, "image"])
    dilated = dilate_image(bw)
    small_ctrs, big_ctrs, mask = filter_contours(dilated, img.shape, 0.006)
    big_ctrs_sorted = sorted(big_ctrs, key=lambda c: cv2.boundingRect(c)[0])
    tokens          = train_set["label"][row_idx].split()

    if len(tokens) == len(big_ctrs_sorted):
        pairs = extract_words_with_labels(img, mask, big_ctrs_sorted, tokens)
        base = os.path.splitext(train_set["filename"][row_idx])[0]
        df = build_word_dataset(pairs, base_name=base)
        visualize_results(img, small_ctrs, big_ctrs_sorted, mask, tokens)
    else:
        print("Mismatch: contours =", len(big_ctrs_sorted), "tokens =", len(tokens))

def filter_contours(dilated: np.ndarray, img_shape: tuple, min_area_ratio: float = 0.001):
    """
    Filters contours by area, removing small ones considered noise.
    Returns:
        - small_contours: list of contours smaller than the area threshold
        - big_contours: list of remaining valid contours
        - cleaned_mask: binary mask (0 = background, 1 = text) with small contours removed
    """

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img_shape[0] * img_shape[1]
    min_area = img_area * min_area_ratio

    cleaned = dilated.copy()
    small_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            small_contours.append(cnt)
            cv2.drawContours(cleaned, [cnt], -1, 0, thickness=cv2.FILLED)

    big_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = cleaned // 255
    
    return small_contours, big_contours, cleaned_mask


def build_word_dataset(word_pairs, base_name):
    """
    word_pairs : [(img_ndarray, label_str), ...]  — те, що вертає extract_words_with_labels
    base_name  : лишив для сумісності, але тут не використовується

    Повертає DataFrame:
      • 'image' — ndarray  (N, H, W)
      • 'label' — текстове слово
    """
    images, labels = zip(*word_pairs)          
    data = {
        "image": list(images),               
        "label": list(labels)
    }
    return pd.DataFrame(data)


def visualize_results(img: np.ndarray,
                      small_contours: list,
                      big_contours: list,
                      cleaned_mask: np.ndarray,
                      tokens: list | None = None,
                      font_scale: float = 0.55,
                      thickness: int = 1):
    """
    Показує:
      • зелений прямокутник – велика контур-«слово»
      • білий заливкою      – маленькі шуми
      • (опційно) текст токена над кожним словом
    """
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for idx, cnt in enumerate(big_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if tokens is not None and idx < len(tokens):
            label = tokens[idx]
            (text_w, text_h), baseline = cv2.getTextSize(label,
                                                         cv2.FONT_HERSHEY_SIMPLEX,
                                                         font_scale, thickness)
            text_x = x
            text_y = max(0, y - 3)
            cv2.putText(vis, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    for cnt in small_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(vis, (x, y), (x + w, y + h),
                      (255, 255, 255), thickness=cv2.FILLED)

    plt.figure(figsize=(12, 4))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title('Detected words (green) with tokens')
    plt.axis('off')
    projection = np.sum(cleaned_mask, axis=0)
    plt.figure(figsize=(10, 3))
    plt.plot(projection)
    plt.title('Vertical Projection (cleaned)')
    plt.xlabel('X (column)')
    plt.ylabel('Count of white pixels')
    plt.grid(True)
    plt.show()

def extract_words_with_labels(img, mask, big_contours, tokens):
    """
    Повертає список (word_img, label_text). Контури мають бути відсортовані.
    """
    word_pairs = []
    for cnt, label in zip(big_contours, tokens):
        x, y, w, h = cv2.boundingRect(cnt)
        crop = img[y:y+h, x:x+w]
        crop_mask = mask[y:y+h, x:x+w]
        crop[crop_mask == 0] = 255   
        word_pairs.append((crop, label))
    return word_pairs


def build_full_word_dataframe(train_set , output_dir):
    word_dfs = []                      

    for row_idx in range(len(train_set)):
        img, bw = load_and_binarize(train_set.loc[row_idx, "image"])
        dilated = dilate_image(bw)
        small_ctrs, big_ctrs, mask = filter_contours(dilated, img.shape, 0.006)

        big_ctrs_sorted = sorted(big_ctrs, key=lambda c: cv2.boundingRect(c)[0])
        tokens = train_set["label"][row_idx].split()

        if len(tokens) == len(big_ctrs_sorted):
            pairs = extract_words_with_labels(img, mask, big_ctrs_sorted, tokens)
            base  = os.path.splitext(train_set["filename"][row_idx])[0]
            df    = build_word_dataset(pairs, base_name=base)
            word_dfs.append(df)
        else:
            print(f"row {row_idx}: contours={len(big_ctrs_sorted)} tokens={len(tokens)}")

    words_df = pd.concat(word_dfs, ignore_index=True)  
    words_df.to_csv(output_dir, index=False) 
    print(f"Done : {len(words_df)} rows in words_df / train_words.csv")
    return words_df
