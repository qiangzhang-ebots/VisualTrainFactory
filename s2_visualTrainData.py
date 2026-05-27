
from pathlib import Path

import cv2


IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def load_yolo_labels(label_path):
    labels = []
    label_file = Path(label_path)
    if not label_file.exists():
        return labels

    with label_file.open('r', encoding='utf-8-sig') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                cls, x, y, w, h = map(float, parts[:5])
            except ValueError:
                continue

            raw_kps = []
            if len(parts) > 5:
                for value in parts[5:]:
                    try:
                        raw_kps.append(float(value))
                    except ValueError:
                        raw_kps.append(0.0)

            keypoints = []
            for i in range(0, len(raw_kps) - 2, 3):
                keypoints.append([raw_kps[i], raw_kps[i + 1]])

            labels.append((int(cls), x, y, w, h, keypoints))

    return labels


def draw_yolo_boxes(img, labels, color=(0, 255, 0)):
    h, w = img.shape[:2]
    for label in labels:
        if len(label) == 6:
            cls, x, y, bw, bh, keypoints = label
        else:
            cls, x, y, bw, bh = label
            keypoints = []

        cx, cy, bw, bh = x * w, y * h, bw * w, bh * h
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, str(cls), (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        if keypoints:
            pts = []
            for kp in keypoints:
                if len(kp) != 2:
                    continue
                kx, ky = kp
                px, py = int(kx * w), int(ky * h)
                cv2.circle(img, (px, py), 3, (0, 0, 255), -1)
                pts.append((px, py))

            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 1)
            if len(pts) > 1:
                cv2.line(img, pts[-1], pts[0], (0, 0, 255), 1)
            if len(keypoints) > 0 and len(keypoints[0]) == 2:
                cv2.circle(img, (int(keypoints[0][0] * w), int(keypoints[0][1] * h)), 10, (255, 0, 0), -1)
            if len(keypoints) > 1 and len(keypoints[1]) == 2:
                cv2.circle(img, (int(keypoints[1][0] * w), int(keypoints[1][1] * h)), 10, (255, 255, 0), -1)
    return img


def visual_Yolo_trainData(img_path, txt_path):
    img_file = Path(img_path)
    if not img_file.exists():
        return None

    img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
    if img is None:
        return None

    labels = load_yolo_labels(txt_path)
    if labels:
        img = draw_yolo_boxes(img, labels)
    return img


def visual_HRNet_trainData(path):
    return None


if __name__ == '__main__':
    pass

