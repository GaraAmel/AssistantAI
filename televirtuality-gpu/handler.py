import os
import uuid
import base64
import tempfile
import traceback

import runpod

from modules.vision_pipeline import (
    run_vision_pipeline
)

from modules.audio_pipeline import (
    transcribe_audio
)


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def save_base64_file(
    base64_data,
    extension
):

    # Permet également :
    # data:image/jpeg;base64,...
    if "," in base64_data:
        base64_data = (
            base64_data.split(",", 1)[1]
        )

    try:

        file_bytes = (
            base64.b64decode(
                base64_data
            )
        )

    except Exception as exc:

        raise ValueError(
            "Fichier Base64 invalide."
        ) from exc

    filename = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    file_path = os.path.join(
        tempfile.gettempdir(),
        filename
    )

    with open(
        file_path,
        "wb"
    ) as f:

        f.write(file_bytes)

    return file_path


def remove_file(path):

    if (
        path
        and
        os.path.exists(path)
    ):

        try:
            os.remove(path)
        except Exception:
            pass


# ============================================================
# HANDLER RUNPOD
# ============================================================

def handler(job):

    temp_file = None

    try:

        job_input = job.get(
            "input",
            {}
        )

        task = job_input.get(
            "task"
        )

        if not task:

            return {
                "success": False,
                "error":
                    "Le champ 'task' est obligatoire."
            }

        # ====================================================
        # VISION IMAGE
        # ====================================================

        if task == "image":

            file_data = job_input.get(
                "file"
            )

            extension = job_input.get(
                "extension",
                ".jpg"
            )

            if not file_data:

                return {
                    "success": False,
                    "error":
                        "Le champ 'file' est obligatoire."
                }

            temp_file = save_base64_file(
                file_data,
                extension
            )

            (
                detections,
                output_path,
                metrics
            ) = run_vision_pipeline(
                temp_file
            )

            return {
                "success": True,
                "task": "image",
                "detections": detections,
                "metrics": metrics
            }

        # ====================================================
        # VIDEO
        # ====================================================

        elif task == "video":

            file_data = job_input.get(
                "file"
            )

            extension = job_input.get(
                "extension",
                ".mp4"
            )

            frame_step = int(
                job_input.get(
                    "frame_step",
                    10
                )
            )

            if frame_step < 1:
                frame_step = 1

            if not file_data:

                return {
                    "success": False,
                    "error":
                        "Le champ 'file' est obligatoire."
                }

            temp_file = save_base64_file(
                file_data,
                extension
            )

            (
                detections,
                output_path,
                metrics
            ) = run_vision_pipeline(
                temp_file,
                frame_step=frame_step
            )

            return {
                "success": True,
                "task": "video",
                "detections": detections,
                "metrics": metrics
            }

        # ====================================================
        # AUDIO
        # ====================================================

        elif task == "audio":

            file_data = job_input.get(
                "file"
            )

            extension = job_input.get(
                "extension",
                ".mp3"
            )

            language = job_input.get(
                "language",
                "fr"
            )

            if not file_data:

                return {
                    "success": False,
                    "error":
                        "Le champ 'file' est obligatoire."
                }

            temp_file = save_base64_file(
                file_data,
                extension
            )

            result = transcribe_audio(
                temp_file,
                language=language
            )

            return {
                "success": True,
                "task": "audio",
                "transcription":
                    result["text"],
                "metrics":
                    result["metrics"]
            }

        # ====================================================
        # TASK INCONNUE
        # ====================================================

        else:

            return {
                "success": False,
                "error":
                    f"Task inconnue : {task}",
                "supported_tasks": [
                    "image",
                    "video",
                    "audio"
                ]
            }

    except Exception as exc:

        traceback.print_exc()

        return {
            "success": False,
            "error": str(exc),
            "type":
                type(exc).__name__
        }

    finally:

        remove_file(
            temp_file
        )


# ============================================================
# DEMARRAGE RUNPOD SERVERLESS
# ============================================================

runpod.serverless.start({
    "handler": handler
})