import os
import time
import torch
import whisper


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "small"
)


print("=" * 60)
print("Initialisation Whisper")
print(f"Device : {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(
    f"Chargement Whisper : "
    f"{WHISPER_MODEL}"
)


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

model = whisper.load_model(
    WHISPER_MODEL,
    device=DEVICE
)

print("Whisper chargé.")
print("=" * 60)


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe_audio(
    audio_path,
    language="fr"
):

    if not os.path.exists(audio_path):

        raise FileNotFoundError(
            f"Fichier audio introuvable : "
            f"{audio_path}"
        )

    start_time = time.perf_counter()

    result = model.transcribe(
        audio_path,
        language=language,
        fp16=torch.cuda.is_available()
    )

    processing_time = (
        time.perf_counter()
        - start_time
    )

    text = result.get(
        "text",
        ""
    ).strip()

    metrics = {

        "processing_time_seconds":
            round(
                processing_time,
                3
            ),

        "device":
            DEVICE,

        "model":
            WHISPER_MODEL
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

    return {
        "text": text,
        "metrics": metrics
    }