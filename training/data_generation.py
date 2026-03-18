"""
Data Generation Script - Surgical Instrument Coating Monitor
Generates synthetic coating images and environmental sensor sequences.

Usage:
    python data_generation.py

Output:
    - data/coating_images/good/        (1000 images)
    - data/coating_images/degraded/    (1000 images)
    - data/coating_images/failed/      (1000 images)
    - data/X_sensor.npy
    - data/y_sensor.npy
    - data/image_paths.npy
    - data/image_labels.npy
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_DIR  = 'data'
IMAGE_DIR = 'data/coating_images'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)


# ─────────────────────────────────────────────
# IMAGE GENERATION
# ─────────────────────────────────────────────
def generate_base_coating(size=224):
    base_color = np.random.randint(160, 200)
    img        = np.ones((size, size, 3), dtype=np.uint8) * base_color
    noise      = np.random.normal(0, 15, img.shape).astype(np.int16)
    img        = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for i in range(size):
        shift  = int(12 * np.sin(i / size * np.pi * np.random.uniform(0.8, 1.2)))
        img[i] = np.clip(img[i].astype(np.int16) + shift, 0, 255)
    for _ in range(np.random.randint(3, 8)):
        cx, cy    = np.random.randint(0, size), np.random.randint(0, size)
        r         = np.random.randint(5, 30)
        intensity = np.random.randint(-12, 12)
        mask      = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        img[:, :, 0] = np.clip(
            img[:, :, 0].astype(np.int16) + (mask > 0) * intensity, 0, 255
        ).astype(np.uint8)
    return img


def add_cracks(img):
    img  = img.copy()
    h, w = img.shape[:2]
    for _ in range(np.random.randint(1, 3)):
        x1, y1      = np.random.randint(0, w), np.random.randint(0, h)
        length      = np.random.randint(10, 40)
        angle       = np.random.uniform(0, np.pi)
        x2          = int(x1 + length * np.cos(angle))
        y2          = int(y1 + length * np.sin(angle))
        crack_color = np.random.randint(120, 155)
        cv2.line(img, (x1, y1), (x2, y2), (crack_color, crack_color, crack_color), 1)
        region_x1 = max(0, min(x1, x2) - 3)
        region_y1 = max(0, min(y1, y2) - 3)
        region_x2 = min(w, max(x1, x2) + 3)
        region_y2 = min(h, max(y1, y2) + 3)
        if region_x2 > region_x1 and region_y2 > region_y1:
            region = img[region_y1:region_y2, region_x1:region_x2]
            img[region_y1:region_y2, region_x1:region_x2] = cv2.GaussianBlur(region, (3, 3), 0)
    return img


def add_discoloration(img):
    img  = img.copy()
    h, w = img.shape[:2]
    for _ in range(np.random.randint(1, 3)):
        cx, cy  = np.random.randint(30, w-30), np.random.randint(30, h-30)
        radius  = np.random.randint(8, 20)
        c_shift = np.random.choice([-1, 1]) * np.random.randint(10, 25)
        mask    = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        mask_blurred = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0) / 255.0
        img[:, :, 1] = np.clip(
            img[:, :, 1].astype(np.float32) + mask_blurred * c_shift, 0, 255
        ).astype(np.uint8)
        img[:, :, 2] = np.clip(
            img[:, :, 2].astype(np.float32) + mask_blurred * (c_shift * 0.4), 0, 255
        ).astype(np.uint8)
    return img


def add_severe_damage(img):
    img  = img.copy()
    h, w = img.shape[:2]
    for _ in range(np.random.randint(1, 3)):
        x, y        = np.random.randint(0, w-40), np.random.randint(0, h-40)
        pw, ph      = np.random.randint(10, 35), np.random.randint(10, 35)
        actual_pw   = min(pw, w - x)
        actual_ph   = min(ph, h - y)
        patch_color = np.random.randint(130, 160, (actual_ph, actual_pw, 3))
        patch_noise = np.random.normal(0, 8, patch_color.shape).astype(np.int16)
        patch_color = np.clip(patch_color.astype(np.int16) + patch_noise, 0, 255).astype(np.uint8)
        img[y:y+actual_ph, x:x+actual_pw] = patch_color
    return add_cracks(img)


def add_lighting_variation(img):
    img       = img.copy()
    h, w      = img.shape[:2]
    cx, cy    = np.random.randint(0, w), np.random.randint(0, h)
    radius    = np.random.randint(40, 100)
    intensity = np.random.choice([-1, 1]) * np.random.randint(8, 20)
    mask      = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    for c in range(3):
        img[:, :, c] = np.clip(
            img[:, :, c].astype(np.float32) + mask * intensity, 0, 255
        ).astype(np.uint8)
    return img


def add_sensor_noise(img):
    img  = img.copy()
    prob = np.random.uniform(0.001, 0.008)
    rnd  = np.random.random(img.shape[:2])
    img[rnd < prob]     = 0
    img[rnd > 1 - prob] = 255
    return img


def generate_coating_dataset(output_dir, n_per_class=1000, size=224, seed=42):
    np.random.seed(seed)
    output_dir = Path(output_dir)
    classes    = ['good', 'degraded', 'failed']

    def make_good(img):
        return add_sensor_noise(add_lighting_variation(img))

    def make_degraded(img):
        return add_sensor_noise(add_lighting_variation(add_discoloration(add_cracks(img))))

    def make_failed(img):
        return add_sensor_noise(add_lighting_variation(add_severe_damage(add_discoloration(img))))

    generators  = {'good': make_good, 'degraded': make_degraded, 'failed': make_failed}
    image_paths = []
    labels      = []

    for cls_idx, cls in enumerate(classes):
        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            base = generate_base_coating(size)
            img  = generators[cls](base)
            path = cls_dir / f"{cls}_{i:04d}.png"
            cv2.imwrite(str(path), img)
            image_paths.append(str(path))
            labels.append(cls_idx)
        print(f"Generated {n_per_class} images for class: {cls}")

    return image_paths, labels


# ─────────────────────────────────────────────
# SENSOR GENERATION
# ─────────────────────────────────────────────
def generate_sensor_reading(timesteps=50, state='stable'):
    t    = np.linspace(0, 1, timesteps)
    base = np.array([22.0, 45.0, 7.0, 0.85, 0.92])
    data = np.tile(base, (timesteps, 1))
    noise_level = 0.05
    if state == 'warning':
        data += np.outer(t, np.array([0.8, 3.0, 0.3, -0.12, -0.08]))
        noise_level = 0.08
    elif state == 'critical':
        data += np.outer(t, np.array([2.5, 8.0, 0.8, -0.35, -0.25]))
        noise_level = 0.15
        spike_idx   = np.random.choice(timesteps, size=5, replace=False)
        data[spike_idx] *= np.random.uniform(1.2, 1.5)
    data += np.random.normal(0, noise_level, data.shape)
    mins = np.array([18.0, 30.0, 5.5, 0.0, 0.0])
    maxs = np.array([35.0, 80.0, 9.0, 1.5, 1.5])
    return np.clip((data - mins) / (maxs - mins), 0, 1).astype(np.float32)


def generate_sensor_dataset(n_per_class=1000, timesteps=50, seed=42):
    np.random.seed(seed)
    states = ['stable', 'warning', 'critical']
    X, y   = [], []
    for cls_idx, state in enumerate(states):
        for _ in range(n_per_class):
            X.append(generate_sensor_reading(timesteps=timesteps, state=state))
            y.append(cls_idx)
        print(f"Generated {n_per_class} sequences for state: {state}")
    X, y = np.array(X), np.array(y)
    idx  = np.random.permutation(len(y))
    return X[idx], y[idx]


# ─────────────────────────────────────────────
# FUSION LABEL LOGIC
# ─────────────────────────────────────────────
def derive_fusion_labels(cnn_labels, lstm_labels):
    labels = []
    for c, s in zip(cnn_labels, lstm_labels):
        c, s = int(c), int(s)
        if   c == 0 and s == 0: labels.append(0)
        elif c == 0 and s == 1: labels.append(1)
        elif c == 1 and s == 0: labels.append(1)
        elif c == 1 and s == 1: labels.append(1)
        elif c == 0 and s == 2: labels.append(2)
        elif c == 2 and s == 0: labels.append(2)
        elif c == 1 and s == 2: labels.append(3)
        elif c == 2 and s == 1: labels.append(3)
        elif c == 2 and s == 2: labels.append(3)
        else:                   labels.append(1)
    return np.array(labels)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':

    # Generate coating images
    print("=" * 50)
    print("Generating coating images...")
    print("=" * 50)
    image_paths, labels = generate_coating_dataset(IMAGE_DIR, n_per_class=1000, seed=42)
    labels              = np.array(labels)
    print(f"\nTotal images       : {len(image_paths)}")
    print(f"Class distribution : {np.bincount(labels)}  [Good | Degraded | Failed]")

    # Preview grid
    classes = ['good', 'degraded', 'failed']
    colors  = ['#4CAF50', '#FF9800', '#F44336']
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle('Synthetic Coating Images', fontsize=14, fontweight='bold')
    for row, (cls, color) in enumerate(zip(classes, colors)):
        files = sorted(Path(IMAGE_DIR).glob(f'{cls}/*.png'))[:4]
        for col, f in enumerate(files):
            img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_title(cls.capitalize(), color=color, fontweight='bold')
    plt.tight_layout()
    plt.savefig('coating_samples.png', dpi=150)
    plt.show()
    print("Sample grid saved to coating_samples.png")

    # Generate sensor sequences
    print("\n" + "=" * 50)
    print("Generating sensor sequences...")
    print("=" * 50)
    X_sensor, y_sensor = generate_sensor_dataset(n_per_class=1000, timesteps=50, seed=42)
    print(f"Sensor data shape  : {X_sensor.shape}")
    print(f"Label distribution : {np.bincount(y_sensor)}  [Stable | Warning | Critical]")

    # Sensor preview
    channel_names = ['Temperature', 'Humidity', 'pH', 'Electrochemical', 'Optical']
    ch_colors     = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    state_names   = ['Stable', 'Warning', 'Critical']
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Sensor Sequences by State', fontsize=14, fontweight='bold')
    for cls_idx, (state, ax) in enumerate(zip(state_names, axes)):
        idx = np.where(y_sensor == cls_idx)[0][0]
        for i, (name, color) in enumerate(zip(channel_names, ch_colors)):
            ax.plot(X_sensor[idx, :, i], label=name, color=color, linewidth=1.5)
        ax.set_title(f'State: {state}', fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig('sensor_samples.png', dpi=150)
    plt.show()
    print("Sensor sample plot saved to sensor_samples.png")

    # Build paired dataset with all 9 CNN/LSTM class combinations
    print("\n" + "=" * 50)
    print("Building paired dataset...")
    print("=" * 50)
    image_labels_arr  = np.array(labels)
    sensor_labels_arr = np.array(y_sensor)
    img_cls = {i: np.where(image_labels_arr == i)[0] for i in range(3)}
    sen_cls = {i: np.where(sensor_labels_arr == i)[0] for i in range(3)}
    rng = np.random.RandomState(42)
    for i in range(3):
        rng.shuffle(img_cls[i])
        rng.shuffle(sen_cls[i])
    n_per_combo  = 333
    pairs_img, pairs_sen = [], []
    for c in range(3):
        for s in range(3):
            img_idx = img_cls[c][:n_per_combo]
            sen_idx = sen_cls[s][:n_per_combo]
            pairs_img.extend(img_idx)
            pairs_sen.extend(sen_idx)
            img_cls[c] = np.roll(img_cls[c], n_per_combo)
            sen_cls[s] = np.roll(sen_cls[s], n_per_combo)
    pairs_img     = np.array(pairs_img)
    pairs_sen     = np.array(pairs_sen)
    final_shuffle = rng.permutation(len(pairs_img))
    pairs_img     = pairs_img[final_shuffle]
    pairs_sen     = pairs_sen[final_shuffle]

    image_paths_final  = np.array(image_paths)[pairs_img]
    image_labels_final = image_labels_arr[pairs_img]
    X_sensor_final     = X_sensor[pairs_sen]
    y_sensor_final     = sensor_labels_arr[pairs_sen]

    # Save
    np.save(f'{DATA_DIR}/X_sensor.npy',     X_sensor_final)
    np.save(f'{DATA_DIR}/y_sensor.npy',     y_sensor_final)
    np.save(f'{DATA_DIR}/image_paths.npy',  image_paths_final)
    np.save(f'{DATA_DIR}/image_labels.npy', image_labels_final)

    # Preview fusion labels
    fusion_preview = derive_fusion_labels(image_labels_final, y_sensor_final)
    labels_map     = {0: 'All Clear', 1: 'Monitor', 2: 'Alert', 3: 'Critical'}
    print("\nImage label distribution  :", np.bincount(image_labels_final))
    print("Sensor label distribution :", np.bincount(y_sensor_final))
    print("\nFusion label distribution (preview):")
    for label, count in sorted(Counter(fusion_preview).items()):
        pct = count / len(fusion_preview) * 100
        print(f"  {labels_map[label]:12} : {count} ({pct:.1f}%)")

    print("\nDone. Run cnn_training.py next.")