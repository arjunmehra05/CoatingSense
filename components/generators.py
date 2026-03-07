import numpy as np
import cv2


def generate_base_coating(size=224):
    base_color = np.random.randint(155, 205)
    img = np.ones((size, size, 3), dtype=np.uint8) * base_color
    for ch in range(3):
        img[:, :, ch] = np.clip(img[:, :, ch].astype(np.int16) + np.random.randint(-15, 15), 0, 255)
    noise = np.random.normal(0, 18, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
    for i in range(size):
        shift = int(15 * np.sin(i / size * np.pi * np.random.uniform(0.5, 2.0)))
        if direction == 'horizontal':
            img[i, :] = np.clip(img[i, :].astype(np.int16) + shift, 0, 255)
        elif direction == 'vertical':
            img[:, i] = np.clip(img[:, i].astype(np.int16) + shift, 0, 255)
        else:
            pixel = img[i, i % size].astype(np.int16)
            img[i, i % size] = np.clip(pixel + shift, 0, 255).astype(np.uint8)
    for _ in range(np.random.randint(4, 10)):
        cx, cy = np.random.randint(0, size), np.random.randint(0, size)
        r = np.random.randint(3, 35)
        intensity = np.random.randint(-18, 18)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        ch = np.random.randint(0, 3)
        img[:, :, ch] = np.clip(
            img[:, :, ch].astype(np.int16) + (mask > 0) * intensity, 0, 255
        ).astype(np.uint8)
    if np.random.random() < 0.3:
        ksize = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img


def add_cracks(img):
    img = img.copy()
    h, w = img.shape[:2]
    if np.random.random() < 0.2:
        return img
    for _ in range(np.random.randint(1, 3)):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        length = np.random.randint(8, 35)
        angle = np.random.uniform(0, np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        base_intensity = int(np.mean(img[max(0, y1-2):y1+2, max(0, x1-2):x1+2]))
        crack_color = int(np.clip(base_intensity + np.random.randint(-25, -10), 0, 255))
        cv2.line(img, (x1, y1), (x2, y2), (crack_color, crack_color, crack_color), 1)
        rx1, ry1 = max(0, min(x1, x2)-3), max(0, min(y1, y2)-3)
        rx2, ry2 = min(w, max(x1, x2)+3), min(h, max(y1, y2)+3)
        if rx2 > rx1 and ry2 > ry1:
            region = img[ry1:ry2, rx1:rx2]
            img[ry1:ry2, rx1:rx2] = cv2.GaussianBlur(region, (3, 3), 0)
    return img


def add_discoloration(img):
    img = img.copy()
    h, w = img.shape[:2]
    if np.random.random() < 0.15:
        return img
    for _ in range(np.random.randint(1, 3)):
        cx, cy = np.random.randint(20, w-20), np.random.randint(20, h-20)
        radius = np.random.randint(6, 18)
        c_shift = np.random.choice([-1, 1]) * np.random.randint(8, 20)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        mask_b = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 0) / 255.0
        img[:, :, 1] = np.clip(
            img[:, :, 1].astype(np.float32) + mask_b * c_shift, 0, 255
        ).astype(np.uint8)
        img[:, :, 2] = np.clip(
            img[:, :, 2].astype(np.float32) + mask_b * (c_shift * 0.3), 0, 255
        ).astype(np.uint8)
    return img


def add_severe_damage(img):
    img = img.copy()
    h, w = img.shape[:2]
    for _ in range(np.random.randint(2, 5)):
        x, y = np.random.randint(0, w-40), np.random.randint(0, h-40)
        pw, ph = np.random.randint(15, 45), np.random.randint(15, 45)
        apw, aph = min(pw, w-x), min(ph, h-y)
        patch = np.random.randint(80, 130, (aph, apw, 3))
        patch_n = np.random.normal(0, 12, patch.shape).astype(np.int16)
        patch = np.clip(patch.astype(np.int16) + patch_n, 0, 255).astype(np.uint8)
        img[y:y+aph, x:x+apw] = patch
    for _ in range(np.random.randint(2, 5)):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        length = np.random.randint(20, 60)
        angle = np.random.uniform(0, np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        crack_color = int(np.clip(np.random.randint(40, 80), 0, 255))
        cv2.line(img, (x1, y1), (x2, y2), (crack_color, crack_color, crack_color), 2)
    return img


def add_lighting_variation(img):
    img = img.copy()
    h, w = img.shape[:2]
    for _ in range(np.random.randint(1, 3)):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(30, 110)
        intensity = np.random.choice([-1, 1]) * np.random.randint(6, 22)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        for c in range(3):
            img[:, :, c] = np.clip(
                img[:, :, c].astype(np.float32) + mask * intensity, 0, 255
            ).astype(np.uint8)
    return img


def add_sensor_noise_img(img):
    img = img.copy()
    prob = np.random.uniform(0.001, 0.01)
    rnd = np.random.random(img.shape[:2])
    img[rnd < prob] = 0
    img[rnd > 1 - prob] = 255
    return img


def add_random_rotation(img):
    h, w = img.shape[:2]
    angle = np.random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def generate_coating_image(coating_state):
    base = generate_base_coating(224)
    if coating_state == 'degraded':
        img = add_discoloration(add_cracks(base))
        img = add_lighting_variation(img)
        img = add_sensor_noise_img(img)
        img = add_random_rotation(img)
    elif coating_state == 'failed':
        img = add_severe_damage(add_discoloration(base))
        img = add_lighting_variation(img)
        img = add_sensor_noise_img(img)
        img = add_random_rotation(img)
    else:
        img = add_lighting_variation(base)
        img = add_sensor_noise_img(img)
        img = add_random_rotation(img)
    return img


def generate_sensor_reading(timesteps=50, state='stable'):
    t = np.linspace(0, 1, timesteps)
    base = np.array([22.0, 45.0, 7.0, 0.85, 0.92])
    base += np.random.normal(0, 0.3, base.shape)
    data = np.tile(base, (timesteps, 1))
    noise_level = 0.05
    if state == 'warning':
        drift = np.array([1.2, 4.0, 0.4, -0.15, -0.10])
        drift += np.random.normal(0, 0.05, drift.shape)
        data += np.outer(t, drift)
        noise_level = 0.08
    elif state == 'critical':
        drift = np.array([2.5, 8.0, 0.8, -0.35, -0.25])
        drift += np.random.normal(0, 0.1, drift.shape)
        data += np.outer(t, drift)
        noise_level = 0.15
        spike_idx = np.random.choice(timesteps, size=np.random.randint(3, 6), replace=False)
        data[spike_idx] *= np.random.uniform(1.2, 1.5)
    data += np.random.normal(0, noise_level, data.shape)
    mins = np.array([18.0, 30.0, 5.5, 0.0, 0.0])
    maxs = np.array([35.0, 80.0, 9.0, 1.5, 1.5])
    return np.clip((data - mins) / (maxs - mins), 0, 1).astype(np.float32)