import json
import os
import re
import tempfile
import time
from enum import Enum
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from modules.audio_pipeline import transcribe_audio
from modules.llm_pipeline import generate_suggestions
from modules.rag_pipeline import retrieve_context
from modules.vision_pipeline import run_vision_pipeline


# =========================================================
# Valeurs proposées dans Swagger
# =========================================================

class ProfilPublic(str, Enum):
    enfants = "enfants"
    scolaires = "scolaires"
    adultes = "adultes"
    experts = "experts"


class ScenarioVisite(str, Enum):
    zoo = "zoo"
    safari = "safari"
    aquarium = "aquarium"
    ferme = "ferme"
    foret = "forêt"
    reserve_naturelle = "réserve naturelle"
    autre = "autre"


# =========================================================
# Application FastAPI
# =========================================================

app = FastAPI(
    title="TeleVirtuality Assistant IA API",
    description=(
        "API de l'assistant IA TeleVirtuality : vision, audio, "
        "RAG et génération de suggestions pour le médiateur."
    ),
    version="2.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://televirtuality.vercel.app",
        "https://api.televirtuality-ai.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# API Key
# =========================================================

API_KEY = os.getenv("API_KEY")


def verify_api_key(x_api_key: Optional[str]) -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API_KEY non configurée sur le serveur.",
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Clé API invalide.",
        )


# =========================================================
# Fonctions utilitaires
# =========================================================

async def save_upload_to_temp(
    upload: UploadFile,
    default_name: str,
) -> str:
    filename = upload.filename or default_name
    extension = os.path.splitext(filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension,
    ) as temp_file:
        content = await upload.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail=f"Le fichier {filename} est vide.",
            )

        temp_file.write(content)
        return temp_file.name


def delete_temp_file(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def extract_section(text: str, title: str) -> str:
    pattern = rf"## {re.escape(title)}\s*(.*?)(?=\n## |\Z)"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    return ""


def parse_assistant_response(raw_response: str) -> dict:
    return {
        "objets_detectes": extract_section(
            raw_response,
            "Objets détectés",
        ),
        "resume_scene": extract_section(
            raw_response,
            "Résumé de la scène",
        ),
        "reformulation_mediateur": extract_section(
            raw_response,
            "Reformulation du médiateur",
        ),
        "informations_utiles_par_animal": extract_section(
            raw_response,
            "Informations utiles par animal",
        ),
        "comparaison": extract_section(
            raw_response,
            "Comparaison",
        ),
        "anecdote": extract_section(
            raw_response,
            "Anecdote",
        ),
        "statut_conservation": extract_section(
            raw_response,
            "Statut de conservation",
        ),
        "question_public": extract_section(
            raw_response,
            "Question à poser au public",
        ),
        "conseil_mediateur": extract_section(
            raw_response,
            "Conseil au médiateur",
        ),
        "reponse_spectateur": extract_section(
            raw_response,
            "Réponse à la question du spectateur",
        ),
        "raw_response": raw_response,
    }


def parse_detections_json(detections_json: str) -> list:
    try:
        detections = json.loads(detections_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"detections_json n'est pas un JSON valide : {exc}",
        ) from exc

    if not isinstance(detections, list):
        raise HTTPException(
            status_code=400,
            detail="detections_json doit contenir une liste JSON.",
        )

    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"La détection à l'indice {index} "
                    "doit être un objet JSON."
                ),
            )

        animal = detection.get("animal") or detection.get("label")
        if not animal:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"La détection à l'indice {index} "
                    "ne contient ni 'animal' ni 'label'."
                ),
            )

    return detections


async def get_transcription(
    audio_file: Optional[UploadFile],
    transcription_text: str = "",
) -> tuple[str, Optional[str]]:
    audio_path = None

    if audio_file is not None:
        audio_path = await save_upload_to_temp(
            audio_file,
            "audio.wav",
        )

        transcription = await run_in_threadpool(
            transcribe_audio,
            audio_path,
        )

        return transcription, audio_path

    if transcription_text and transcription_text.strip():
        return transcription_text.strip(), None

    return "Aucun audio fourni.", None


async def generate_structured_assistant(
    detections: list,
    transcription: str,
    profil_public: str,
    scenario: str,
    spectator_question: str,
) -> tuple[dict, bool]:
    rag_context = await run_in_threadpool(
        retrieve_context,
        detections,
    )

    raw_response = await run_in_threadpool(
        generate_suggestions,
        detections,
        transcription,
        profil_public,
        rag_context,
        scenario,
        spectator_question,
    )

    structured_response = parse_assistant_response(raw_response)

    return structured_response, bool(rag_context)


# =========================================================
# Routes générales
# =========================================================

@app.get("/")
def home():
    return {
        "message": "API Assistant IA TeleVirtuality fonctionne.",
        "version": "2.0.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "API en ligne",
    }


@app.get("/ping")
def ping():
    return {
        "status": "ok",
        "message": "Assistant IA accessible",
    }


# =========================================================
# 1. Endpoint Vision — rapide pour les bounding boxes
# =========================================================

@app.post("/vision")
async def vision(
    media_file: UploadFile = File(
        ...,
        description="Frame image ou média à analyser",
    ),
    x_api_key: Optional[str] = Header(
        None,
        description="Clé API TeleVirtuality",
    ),
):
    verify_api_key(x_api_key)

    start_time = time.perf_counter()
    media_path = None

    try:
        media_path = await save_upload_to_temp(
            media_file,
            "frame.jpg",
        )

        detections, _ = await run_in_threadpool(
            run_vision_pipeline,
            media_path,
        )

        processing_time = round(
            time.perf_counter() - start_time,
            3,
        )

        return {
            "status": "success",
            "endpoint": "vision",
            "processing_time_seconds": processing_time,
            "vision": {
                "num_detections": len(detections),
                "detections": detections,
            },
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant l'analyse visuelle : {exc}",
        ) from exc

    finally:
        delete_temp_file(media_path)


# =========================================================
# 2. Endpoint Assistant — RAG + LLM sans relancer la vision
# =========================================================

@app.post("/assistant")
async def assistant(
    detections_json: str = Form(
        ...,
        description=(
            "Liste JSON des détections obtenues depuis /vision"
        ),
    ),
    profil_public: ProfilPublic = Form(
        ...,
        description="Profil du public cible",
    ),
    scenario: ScenarioVisite = Form(
        ScenarioVisite.autre,
        description="Scénario ou contexte de la visite",
    ),
    spectator_question: str = Form(
        "",
        description="Question éventuelle d'un spectateur",
    ),
    transcription: str = Form(
        "",
        description=(
            "Transcription déjà disponible. Ignorée si un "
            "fichier audio est fourni."
        ),
    ),
    audio_file: Optional[UploadFile] = File(
        None,
        description="Segment audio optionnel du médiateur",
    ),
    x_api_key: Optional[str] = Header(
        None,
        description="Clé API TeleVirtuality",
    ),
):
    verify_api_key(x_api_key)

    start_time = time.perf_counter()
    audio_path = None

    try:
        detections = parse_detections_json(
            detections_json,
        )

        final_transcription, audio_path = await get_transcription(
            audio_file=audio_file,
            transcription_text=transcription,
        )

        structured_assistant, context_found = (
            await generate_structured_assistant(
                detections=detections,
                transcription=final_transcription,
                profil_public=profil_public.value,
                scenario=scenario.value,
                spectator_question=spectator_question,
            )
        )

        processing_time = round(
            time.perf_counter() - start_time,
            3,
        )

        return {
            "status": "success",
            "endpoint": "assistant",
            "processing_time_seconds": processing_time,
            "inputs": {
                "profil_public": profil_public.value,
                "scenario": scenario.value,
                "spectator_question": spectator_question,
                "audio_filename": (
                    audio_file.filename
                    if audio_file
                    else None
                ),
            },
            "vision": {
                "num_detections": len(detections),
                "detections": detections,
            },
            "audio": {
                "transcription": final_transcription,
            },
            "rag": {
                "context_found": context_found,
            },
            "assistant": structured_assistant,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la génération IA : {exc}",
        ) from exc

    finally:
        delete_temp_file(audio_path)


# =========================================================
# 3. Endpoint complet — compatibilité et tests Swagger
# =========================================================

@app.post("/analyze")
async def analyze(
    media_file: UploadFile = File(
        ...,
        description="Image ou vidéo à analyser",
    ),
    audio_file: Optional[UploadFile] = File(
        None,
        description="Fichier audio optionnel du médiateur",
    ),
    profil_public: ProfilPublic = Form(
        ...,
        description="Profil du public cible",
    ),
    scenario: ScenarioVisite = Form(
        ScenarioVisite.autre,
        description="Scénario ou contexte de la visite",
    ),
    spectator_question: str = Form(
        "",
        description="Question éventuelle d'un spectateur",
    ),
    x_api_key: Optional[str] = Header(
        None,
        description="Clé API TeleVirtuality",
    ),
):
    verify_api_key(x_api_key)

    total_start = time.perf_counter()
    media_path = None
    audio_path = None

    try:
        # -----------------------------------------
        # Sauvegarde du média
        # -----------------------------------------

        media_path = await save_upload_to_temp(
            media_file,
            "media",
        )

        # -----------------------------------------
        # Audio
        # -----------------------------------------

        audio_start = time.perf_counter()

        transcription, audio_path = await get_transcription(
            audio_file=audio_file,
        )

        audio_time = round(
            time.perf_counter() - audio_start,
            3,
        )

        # -----------------------------------------
        # Vision
        # -----------------------------------------

        vision_start = time.perf_counter()

        detections, output_path = await run_in_threadpool(
            run_vision_pipeline,
            media_path,
        )

        vision_time = round(
            time.perf_counter() - vision_start,
            3,
        )

        # -----------------------------------------
        # RAG + LLM
        # -----------------------------------------

        assistant_start = time.perf_counter()

        structured_assistant, context_found = (
            await generate_structured_assistant(
                detections=detections,
                transcription=transcription,
                profil_public=profil_public.value,
                scenario=scenario.value,
                spectator_question=spectator_question,
            )
        )

        assistant_time = round(
            time.perf_counter() - assistant_start,
            3,
        )

        total_time = round(
            time.perf_counter() - total_start,
            3,
        )

        # -----------------------------------------
        # Réponse JSON
        # -----------------------------------------

        return {
            "status": "success",
            "endpoint": "analyze",
            "processing_time_seconds": total_time,
            "processing_steps_seconds": {
                "vision": vision_time,
                "audio": audio_time,
                "rag_and_llm": assistant_time,
            },
            "inputs": {
                "media_filename": media_file.filename,
                "audio_filename": (
                    audio_file.filename
                    if audio_file
                    else None
                ),
                "profil_public": profil_public.value,
                "scenario": scenario.value,
                "spectator_question": spectator_question,
            },
            "vision": {
                "num_detections": len(detections),
                "detections": detections,

                # Chemin local utile pour le développement.
                # La Web App ne doit pas utiliser ce chemin.
                "annotated_output_path": output_path,
            },
            "audio": {
                "transcription": transcription,
            },
            "rag": {
                "context_found": context_found,
            },
            "assistant": structured_assistant,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant l'analyse complète : {exc}",
        ) from exc

    finally:
        delete_temp_file(media_path)
        delete_temp_file(audio_path)