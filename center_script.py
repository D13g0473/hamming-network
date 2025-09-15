import os
import csv
import numpy as np

def load_csv(filepath):
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        data = np.array([[int(x) for x in row] for row in reader])
    return data

def save_csv(filepath, data):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

def center_image(data, size=28):
    # Encuentra bounding box
    rows = np.any(data, axis=1)
    cols = np.any(data, axis=0)

    if not rows.any() or not cols.any():
        return data  # imagen vacía → no tocar

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Recorte
    cropped = data[ymin:ymax+1, xmin:xmax+1]

    h, w = cropped.shape
    new_img = np.zeros((size, size), dtype=int)

    # Calcular offsets para centrar
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    # Pegar en el centro
    new_img[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    return new_img

def process_dataset(base_dir="dataset_test", out_dir="dataset_test_centered"):
    os.makedirs(out_dir, exist_ok=True)

    for shape in os.listdir(base_dir):
        shape_dir = os.path.join(base_dir, shape)
        if not os.path.isdir(shape_dir):
            continue

        out_shape_dir = os.path.join(out_dir, shape)
        os.makedirs(out_shape_dir, exist_ok=True)

        for file in os.listdir(shape_dir):
            if not file.endswith(".csv"):
                continue

            filepath = os.path.join(shape_dir, file)
            data = load_csv(filepath)
            centered = center_image(data)

            outpath = os.path.join(out_shape_dir, file)
            save_csv(outpath, centered)
            print(f"Centrado: {outpath}")

if __name__ == "__main__":
    process_dataset()
