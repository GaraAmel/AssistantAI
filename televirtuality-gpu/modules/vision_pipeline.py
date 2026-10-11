import os
import time
import cv2
import torch

from ultralytics import YOLO


# ============================================================
# CONFIGURATION GPU
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Initialisation du pipeline YOLO")
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA disponible : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
else:
    print("ATTENTION : exécution sur CPU")

print("=" * 60)


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    "models/best.pt"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Modèle YOLO introuvable : {MODEL_PATH}"
    )

print(f"Chargement du modèle YOLO : {MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("Modèle YOLO chargé.")


# ============================================================
# DESSIN DES BOXES
# ============================================================

def draw_boxes(image, detections):

    if image is None:
        return image

    for det in detections:

        x1, y1, x2, y2 = map(int, det["box"])

        label = (
            f'{det["animal"]} '
            f'{det["confidence"]:.2f}'
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return image


# ============================================================
# EXTRACTION DES DETECTIONS YOLO
# ============================================================

def extract_detections(results, frame_index=None):

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])

            label = model.names[class_id]

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            detection = {
                "animal": label,
                "confidence": round(
                    confidence,
                    3
                ),
                "box": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ]
            }

            if frame_index is not None:
                detection["frame"] = frame_index

            detections.append(detection)

    return detections


# ============================================================
# IMAGE
# ============================================================

def predict_image(
    image_path,
    conf=0.25,
    output_dir="outputs"
):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image introuvable : {image_path}"
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    start_time = time.perf_counter()

    results = model.predict(
        source=image_path,
        conf=conf,
        device=DEVICE,
        save=False,
        verbose=False
    )

    inference_time = (
        time.perf_counter() - start_time
    )

    detections = extract_detections(
        results
    )

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Impossible de lire l'image : "
            f"{image_path}"
        )

    annotated = draw_boxes(
        image,
        detections
    )

    output_path = os.path.join(
        output_dir,
        "annotated_image.jpg"
    )

    cv2.imwrite(
        output_path,
        annotated
    )

    metrics = {
        "inference_time_seconds":
            round(inference_time, 4),

        "number_of_detections":
            len(detections),

        "device":
            str(DEVICE)
    }

    if torch.cuda.is_available():

        metrics["gpu"] = (
            torch.cuda.get_device_name(0)
        )

        metrics["vram_allocated_mb"] = round(
            torch.cuda.memory_allocated(0)
            / 1024**2,
            2
        )

        metrics["vram_reserved_mb"] = round(
            torch.cuda.memory_reserved(0)
            / 1024**2,
            2
        )

    return detections, output_path, metrics


# ============================================================
# VIDEO
# ============================================================

def predict_video(
    video_path,
    frame_step=10,
    conf=0.25,
    output_dir="outputs"
):

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Vidéo introuvable : "
            f"{video_path}"
        )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise ValueError(
            f"Impossible d'ouvrir la vidéo : "
            f"{video_path}"
        )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        output_dir,
        "annotated_video.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    all_detections = []
    last_detections = []

    frame_index = 0
    analyzed_frames = 0

    total_inference_time = 0.0

    global_start = time.perf_counter()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % frame_step == 0:

            inference_start = (
                time.perf_counter()
            )

            results = model.predict(
                source=frame,
                conf=conf,
                device=DEVICE,
                save=False,
                verbose=False
            )

            inference_duration = (
                time.perf_counter()
                - inference_start
            )

            total_inference_time += (
                inference_duration
            )

            analyzed_frames += 1

            frame_detections = (
                extract_detections(
                    results,
                    frame_index=frame_index
                )
            )

            all_detections.extend(
                frame_detections
            )

            last_detections = (
                frame_detections
            )

        annotated_frame = draw_boxes(
            frame.copy(),
            last_detections
        )

        writer.write(
            annotated_frame
        )

        frame_index += 1

    cap.release()
    writer.release()

    total_processing_time = (
        time.perf_counter()
        - global_start
    )

    # --------------------------------------------------------
    # Garder la meilleure détection de chaque animal
    # --------------------------------------------------------

    best_detections = {}

    for det in all_detections:

        animal = det["animal"]

        if animal not in best_detections:

            best_detections[animal] = det

        elif (
            det["confidence"]
            >
            best_detections[animal][
                "confidence"
            ]
        ):

            best_detections[animal] = det

    # --------------------------------------------------------
    # METRIQUES
    # --------------------------------------------------------

    if analyzed_frames > 0:

        average_inference = (
            total_inference_time
            / analyzed_frames
        )

        inference_fps = (
            1 / average_inference
            if average_inference > 0
            else 0
        )

    else:

        average_inference = 0
        inference_fps = 0

    video_duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    metrics = {

        "video_duration_seconds":
            round(video_duration, 2),

        "video_fps":
            round(fps, 2),

        "total_frames":
            total_frames,

        "analyzed_frames":
            analyzed_frames,

        "frame_step":
            frame_step,

        "total_processing_time_seconds":
            round(
                total_processing_time,
                3
            ),

        "total_yolo_inference_seconds":
            round(
                total_inference_time,
                3
            ),

        "average_inference_ms":
            round(
                average_inference * 1000,
                2
            ),

        "inference_fps":
            round(
                inference_fps,
                2
            ),

        "device":
            str(DEVICE)
    }

    if torch.cuda.is_available():

        metrics["gpu"] = (
            torch.cuda.get_device_name(0)
        )

        metrics["vram_allocated_mb"] = round(
            torch.cuda.memory_allocated(0)
            / 1024**2,
            2
        )

        metrics["vram_reserved_mb"] = round(
            torch.cuda.memory_reserved(0)
            / 1024**2,
            2
        )

    return (
        list(best_detections.values()),
        output_path,
        metrics
    )


# ============================================================
# PIPELINE GENERAL
# ============================================================

def run_vision_pipeline(
    file_path,
    frame_step=10
):

    ext = os.path.splitext(
        file_path
    )[1].lower()

    if ext in [
        ".jpg",
        ".jpeg",
        ".png"
    ]:

        return predict_image(
            file_path
        )

    elif ext in [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    ]:

        return predict_video(
            file_path,
            frame_step=frame_step
        )

    else:

        raise ValueError(
            f"Format non supporté : {ext}"
        )