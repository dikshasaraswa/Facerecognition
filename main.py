import cv2
import os

def detect_faces_dnn(image_path, output_dir, model_path, config_path, confidence_threshold=0.5):
    # Load model
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found")
    h, w = image.shape[:2]

    # Preprocess for DNN
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = image[y1:y2, x1:x2]
            if face.size == 0:
                continue

            count += 1
            out_path = os.path.join(output_dir, f"face_{count}.jpg")
            cv2.imwrite(out_path, face)
            print(f"Saved: {out_path}")

    print(f"Total faces saved: {count}")
    return count

# Example usage
detect_faces_dnn(
    image_path=r"D:\Git_main\Facerecognition\360_F_326085309_CFH8PpadfnL2OQ7Gi411XW0B21YumxKo.jpg",
    output_dir=r"D:\Git_main\Facerecognition\myfaces",
    model_path=r"D:\Git_main\Facerecognition\res10_300x300_ssd_iter_140000.caffemodel",
    config_path=r"D:\Git_main\Facerecognition\deploy.prototxt"
)
