import pandas as pd
import os
from PIL import Image
import numpy as np
import json

# Mapping for filesystem-safe names of special symbols.
SYMBOL_NAMES = {
    "?": "questionmark",
    '"': "doublequote",
    "/": "forwardslash",
    ":": "colon",
}


def get_dictionary(df: pd.DataFrame) -> set:
    """Return a set of unique characters found in df['label']."""
    letters = set()
    for word in df["label"]:
        for letter in word:
            letters.add(letter)
    return letters


def get_with_letter_dataset(
    letter: str, name_letter: str, word_df: pd.DataFrame, folder_path: str
) -> pd.DataFrame:
    """
    Build a subset where labels contain `letter`, save images under `<folder_path>/with/`,
    and return a DataFrame with file paths.
    """
    with_letter_dataset = word_df[
        word_df["label"].str.contains(letter, regex=False)
    ].copy()
    folder_with_path = os.path.join(folder_path, "with")
    os.makedirs(folder_with_path, exist_ok=True)
    file_names = []
    for idx, row in enumerate(with_letter_dataset.itertuples(), 1):
        try:
            pixels = np.array(row.image, dtype=np.uint8)
            img = Image.fromarray(pixels)
            file_name = f"with_{name_letter}_{idx:05d}.png"
            full_path = os.path.join(folder_with_path, file_name)
            img.save(full_path)
            file_names.append(full_path)
        except Exception as e:
            print(f"Can not save the file {letter}_{idx:05d}.png: {e}")
            continue
    with_letter_dataset["file_name"] = file_names
    with_letter_dataset = with_letter_dataset.drop(columns=["image"])
    return with_letter_dataset


def get_without_letter_dataset(
    letter: str,
    name_letter: str,
    word_df: pd.DataFrame,
    folder_path: str,
    num_of_samples: int,
) -> pd.DataFrame:
    """
    Build a random subset where labels do NOT contain `letter` (size ≈ with-letter count),
    save images under `<folder_path>/without/`.
    """
    df_without = word_df[~word_df["label"].str.contains(letter, regex=False)].copy()
    without_letter_dataset = df_without.sample(
        n=min(num_of_samples, len(df_without)), random_state=42
    ).reset_index(drop=True)
    folder_without_path = os.path.join(folder_path, "without")
    os.makedirs(folder_without_path, exist_ok=True)
    file_names = []
    for idx, row in enumerate(without_letter_dataset.itertuples(), 1):
        try:
            pixels = np.array(row.image, dtype=np.uint8)
            img = Image.fromarray(pixels)
            file_name = f"without_{name_letter}_{idx:05d}.png"
            full_path = os.path.join(folder_without_path, file_name)
            img.save(full_path)
            file_names.append(full_path)
        except Exception as e:
            print(f"Can not save the file {letter}_{idx:05d}.png: {e}")
            continue
    without_letter_dataset["file_name"] = file_names
    without_letter_dataset = without_letter_dataset.drop(columns=["image"])
    return without_letter_dataset


def get_with_without_split(letters: set, output_path: str, word_df: pd.DataFrame):
    """
    For each letter in `letters`, create `<output_path>/letters/<name>_letter/` with:
      - saved images for samples 'with' and 'without' the letter,
      - a JSON manifest listing file paths and labels.
    """
    failed_letters = []
    output_root = os.path.join(output_path, "letters")
    os.makedirs(output_root, exist_ok=True)
    for letter in letters:
        # Use friendly names for special symbols (for folder/file safety).
        name_letter = SYMBOL_NAMES.get(letter, letter)
        try:
            folder_path = os.path.join(output_root, f"{name_letter}_letter")
            with_letter_dataset = get_with_letter_dataset(
                letter, name_letter, word_df, folder_path
            )
            without_letter_dataset = get_without_letter_dataset(
                letter, name_letter, word_df, folder_path, len(with_letter_dataset)
            )
            data = {
                "letter": letter,
                "with": with_letter_dataset[["file_name", "label"]].to_dict(
                    orient="records"
                ),
                "without": without_letter_dataset[["file_name", "label"]].to_dict(
                    orient="records"
                ),
            }
            json_path = os.path.join(folder_path, f"{name_letter}_dataset.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Can not prosseced letter '{letter}': {e}")
            failed_letters.append(letter)
    if failed_letters:
        print("Can not prosseced letters :", failed_letters)
    else:
        print(f"Success! All letters are in the folder {output_root}")
