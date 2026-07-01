"""
utils/response_parser.py
========================
Parser yang robust untuk output mentah LLM → SimulationStep yang tervalidasi.

Pipeline:
  raw string dari LLM
      ↓
  _clean_raw()          — buang markdown fence, leading/trailing noise
      ↓
  _extract_json()       — ekstrak JSON object dari string
      ↓
  _normalize_fields()   — normalisasi alias field (timerAction ↔ timer_action, dll)
      ↓
  SimulationStep(**data) — validasi Pydantic (raises ValidationError jika gagal)
      ↓
  SimulationStep (validated)

Jika semua langkah gagal → kembalikan SimulationStep fallback yang aman.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from models.schemas import (
    BiasType,
    FlagBiasAction,
    FlagBiasData,
    NoAction,
    SimulationStep,
)

logger = logging.getLogger(__name__)


# konstan yang mungkin digunakan llm saat halusinasi
_FIELD_ALIASES: dict[str, str] = {
    # timer 
    "timerAction":   "timerAction",
    "timer_action":  "timerAction",
    "timer":         "timerAction",
    "timer_control": "timerAction",
    # text / narrative (support format baru dari system prompt v2)
    "narrative":        "text",
    "dialogue":         "text",
    "patient_dialogue": "text",
    "response":         "text",
    "message":          "text",
    "reply":            "text",
    "narration":        "text",
    # action
    "whiteboard_actions": "actions",
    "whiteboard":         "actions",
    "commands":           "actions",
    "steps":              "actions",
}

# Node type aliases (LLM kadang hallucinate variant)
_NODE_TYPE_ALIASES: dict[str, str] = {
    "vital_sign":         "finding",
    "vital_signs":        "finding",
    "diagnostic_finding": "finding",
    "lab":                "finding",
    "lab_result":         "finding",
    "lab_finding":        "finding",
    "physical_exam":      "finding",
    "physical_finding":   "finding",
    "risk":               "risk_factor",
    "risks":              "risk_factor",
    "risk_factors":       "risk_factor",
    "symptom_node":       "symptom",
    "diagnosis":          "hypothesis",
    "diagnostic":         "hypothesis",
    "differential":       "hypothesis",
}

# Action type aliases
_ACTION_TYPE_ALIASES: dict[str, str] = {
    # add_node variants
    "addnode":           "add_node",
    "add node":          "add_node",
    "add_node":          "add_node",
    "node":              "add_node",
    # add_edge variants
    "addedge":           "add_edge",
    "add edge":          "add_edge",
    "add_edge":          "add_edge",
    "edge":              "add_edge",
    # trigger_hint variants
    "triggerhint":       "trigger_hint",
    "trigger_hint":      "trigger_hint",
    "hint":              "trigger_hint",
    # flag_bias variants
    "flagbias":          "flag_bias",
    "flag_bias":         "flag_bias",
    "bias":              "flag_bias",
    "flag bias":         "flag_bias",
    # update_hypothesis variants
    "updatehypothesis":  "update_hypothesis",
    "update_hypothesis": "update_hypothesis",
    "update hypothesis": "update_hypothesis",
    "update":            "update_hypothesis",
    # no_action variants
    "noaction":          "no_action",
    "no_action":         "no_action",
    "none":              "no_action",
    "noop":              "no_action",
}

# Bias type aliases
_BIAS_TYPE_ALIASES: dict[str, str] = {
    "prematureClosure":   "premature_closure",
    "premature closure":  "premature_closure",
    "closure":            "premature_closure",
    "anchoringBias":      "anchoring_bias",
    "anchoring bias":     "anchoring_bias",
    "anchoring":          "anchoring_bias",
}


def _extract_first_json_object(s: str) -> str:
    """
    Mencari dan mengekstrak objek JSON valid pertama dari string dengan mencocokkan kurung kurawal {}.
    Mengabaikan tanda kurung di dalam string literal dan menghormati karakter escape.
    """
    start = s.find('{')
    if start == -1:
        return s
    
    count = 0
    in_string = False
    escape = False
    
    for i in range(start, len(s)):
        char = s[i]
        
        if escape:
            escape = False
            continue
            
        if char == '\\':
            escape = True
            continue
            
        if char == '"':
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                count += 1
            elif char == '}':
                count -= 1
                if count == 0:
                    return s[start:i+1]
                    
    return s[start:]


def _clean_raw(raw: str) -> str:
    """
    Bersihkan output mentah LLM dari noise umum.
    - Hapus markdown code fence (```json ... ```)
    - Hapus teks sebelum JSON object pertama dan setelah JSON object terakhir
    - Strip whitespace
    """
    if not raw or not raw.strip():
        raise ValueError("LLM returned empty response")

    cleaned = raw.strip()

    # Hapus markdown fence yang paling umum
    patterns = [
        r'^```(?:json|JSON|Json)?\s*\n?',   # opening fence
        r'\n?```\s*$',                        # closing fence
        r'^`{1,2}(?:json)?',                  # single/double backtick
        r'`{1,2}$',
    ]
    for p in patterns:
        cleaned = re.sub(p, '', cleaned, flags=re.MULTILINE).strip()

    # Ekstrak objek JSON pertama menggunakan kurung kurawal pencocokan
    try:
        cleaned = _extract_first_json_object(cleaned)
    except Exception as e:
        logger.warning(f"Gagal melakukan brace matching JSON extraction: {e}")

    return cleaned


def _extract_json(cleaned: str) -> dict[str, Any]:
    """
    Parse string JSON → dict Python.
    Coba json.loads biasa dulu, fallback ke ast.literal_eval untuk edge case.
    """
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"json.loads gagal: {e}. Mencoba repair...")

    # Repair umum: trailing comma sebelum } atau ]
    repaired = re.sub(r',\s*([}\]])', r'\1', cleaned)
    # Repair: single quotes → double quotes (hanya untuk string values sederhana)
    repaired = re.sub(r"(?<![\\])'", '"', repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e2:
        raise ValueError(
            f"Tidak bisa parse JSON dari LLM response setelah repair: {e2}\n"
            f"Raw (first 500 chars): {cleaned[:500]!r}"
        )


def _normalize_fields(data: dict[str, Any]) -> dict[str, Any]:
    """
    Normalisasi field names dan values yang mungkin berbeda antara
    apa yang kita instruksikan di prompt vs apa yang LLM return.
    """
    normalized = {}

    for key, value in data.items():
        # Normalize top-level field names
        canonical_key = _FIELD_ALIASES.get(key, key)
        normalized[canonical_key] = value

    # Normalize 'text' harus string
    if "text" in normalized and not isinstance(normalized["text"], str):
        normalized["text"] = str(normalized["text"])

    # Normalize 'actions' harus list
    if "actions" not in normalized:
        normalized["actions"] = []
    elif not isinstance(normalized["actions"], list):
        normalized["actions"] = [normalized["actions"]] if normalized["actions"] else []

    # Normalize setiap action dalam list
    normalized["actions"] = [
        _normalize_action(a) for a in normalized["actions"]
        if isinstance(a, dict)
    ]

    # Normalize timerAction
    if "timerAction" in normalized:
        ta = normalized["timerAction"]
        if isinstance(ta, str):
            normalized["timerAction"] = ta.lower()
        elif ta is None:
            pass  # OK
        else:
            normalized["timerAction"] = None

    return normalized


def _normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    """
    Normalisasi satu whiteboard action:
    - type aliases → canonical type
    - data field aliases
    - node type aliases dalam add_node data
    - bias type aliases dalam flag_bias data
    """
    if "type" not in action:
        logger.warning(f"Action tanpa 'type' field, skip: {action}")
        return {"type": "no_action", "data": None}

    # Normalize action type
    raw_type = str(action["type"]).strip().lower()
    canonical_type = _ACTION_TYPE_ALIASES.get(raw_type, raw_type)
    action = {**action, "type": canonical_type}

    # Normalize data berdasarkan action type
    data = action.get("data") or {}

    if canonical_type == "add_node" and isinstance(data, dict):
        # Normalize node type
        if "type" in data:
            raw_node_type = str(data["type"]).strip().lower()
            data["type"] = _NODE_TYPE_ALIASES.get(raw_node_type, raw_node_type)
        # Normalize id: pastikan snake_case
        if "id" in data:
            data["id"] = _to_snake_case(str(data["id"]))
        action["data"] = data

    elif canonical_type == "add_edge" and isinstance(data, dict):
        # Normalize source/target IDs
        for field in ("source", "target"):
            if field in data:
                data[field] = _to_snake_case(str(data[field]))
        # Auto-generate edge id jika tidak ada
        if "id" not in data and "source" in data and "target" in data:
            data["id"] = f"edge_{data['source']}_{data['target']}"
        action["data"] = data

    elif canonical_type == "flag_bias" and isinstance(data, dict):
        # Normalize bias type
        if "bias_type" in data:
            raw_bias = str(data["bias_type"]).strip().lower()
            data["bias_type"] = _BIAS_TYPE_ALIASES.get(raw_bias, raw_bias)
        action["data"] = data

    elif canonical_type == "no_action":
        action["data"] = None

    return action


def _to_snake_case(s: str) -> str:
    """
    Konversi string ke snake_case yang valid untuk node ID.
    Hapus karakter non-alfanumerik, ganti spasi/dash dengan underscore.
    """
    s = s.strip().lower()
    s = re.sub(r'[\s\-]+', '_', s)
    s = re.sub(r'[^a-z0-9_]', '', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s or 'node_unknown'


def _build_fallback(reason: str) -> SimulationStep:
    """
    Kembalikan SimulationStep minimal yang aman jika semua parsing gagal.
    Frontend tetap bisa render ini tanpa crash.
    """
    logger.error(f"Parser fallback triggered: {reason}")
    return SimulationStep(
        text=(
            "Maaf, sistem sedang mengalami gangguan teknis. "
            "Silakan ulangi pertanyaan Anda."
        ),
        actions=[NoAction(type="no_action", data=None)],
        timerAction=None,
    )


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def parse_llm_response(raw: str) -> SimulationStep:
    """
    Entry point utama. Konversi raw LLM string → SimulationStep tervalidasi.

    Args:
        raw: String mentah dari OpenAI ChatCompletion response.

    Returns:
        SimulationStep yang sudah divalidasi Pydantic.
        Tidak pernah raise — jika gagal, kembalikan fallback yang aman.

    Pipeline:
        raw → _clean_raw → _extract_json → _normalize_fields → SimulationStep
    """
    try:
        cleaned = _clean_raw(raw)
    except ValueError as e:
        return _build_fallback(f"clean step failed: {e}")

    try:
        data = _extract_json(cleaned)
    except ValueError as e:
        return _build_fallback(f"json extraction failed: {e}")

    try:
        normalized = _normalize_fields(data)
    except Exception as e:
        logger.warning(f"Normalization error (continuing with raw data): {e}")
        normalized = data  # best-effort

    try:
        step = SimulationStep(**normalized)
        logger.debug(
            f"Parsed SimulationStep: text={step.text[:50]!r}, "
            f"actions={[a.type for a in step.actions]}, "
            f"timer={step.timer_action}"
        )
        return step

    except ValidationError as e:
        # Coba recovery parsial: pertahankan 'text', drop actions yang invalid
        logger.warning(f"ValidationError, mencoba recovery parsial: {e}")
        try:
            safe_data = {
                "text":        normalized.get("text", "Maaf, ada kendala teknis."),
                "actions":     [],  # drop semua actions yang bermasalah
                "timerAction": normalized.get("timerAction"),
            }
            return SimulationStep(**safe_data)
        except ValidationError as e2:
            return _build_fallback(f"Full validation failed: {e2}")


def parse_and_inject_bias(
    raw: str,
    bias_type: str,
    bias_description: str,
    affected_hypothesis: str | None = None,
) -> SimulationStep:
    """
    Parser yang dipanggil HANYA oleh director saat backend mendeteksi bias.
    """
    # 1. Parse hasil mentah dari LLM
    step = parse_llm_response(raw)

    # 2. Buat objek bias yang valid
    try:
        bias_action = FlagBiasAction(
            type="flag_bias",
            data=FlagBiasData(
                bias_type            = BiasType(bias_type),
                description          = bias_description,
                affected_hypothesis  = affected_hypothesis,
            )
        )
    except Exception as e:
        logger.error(f"Gagal membuat FlagBiasAction: {e}")
        return step 

    # --- PERUBAHAN PENTING DI SINI ---
    
    # 3. Filter/Sanitize:
    # Kita pastikan aksi 'flag_bias' dari LLM dibuang total, 
    # karena kita tidak ingin LLM memberikan deteksi bias sendiri (hanya backend).
    step.actions = [a for a in step.actions if a.type != "flag_bias"]

    # 4. Inject bias dari Backend di urutan PALING AWAL
    step.actions.insert(0, bias_action)

    # ---------------------------------

    logger.info(f"Bias {bias_type} berhasil diinjeksi ke SimulationStep oleh Backend.")
    return step

def validate_step(step: SimulationStep) -> tuple[bool, list[str]]:
    """
    Validasi tambahan di luar Pydantic untuk business logic rules.

    Returns:
        (is_valid: bool, warnings: list[str])
        Warnings tidak memblokir, hanya di-log.
    """
    warnings: list[str] = []

    # Cek: edge harus referensi node yang ada di actions yang sama (atau sudah ada di canvas)
    node_ids_in_step = {
        a.data.id
        for a in step.actions
        if a.type == "add_node"
    }
    for action in step.actions:
        if action.type == "add_edge":
            d = action.data
            if d.source not in node_ids_in_step:
                warnings.append(
                    f"Edge '{d.id}' referensi source '{d.source}' yang tidak ada di step ini. "
                    "Pastikan node sudah ada di canvas sebelum edge ditambahkan."
                )
            if d.target not in node_ids_in_step:
                warnings.append(
                    f"Edge '{d.id}' referensi target '{d.target}' yang tidak ada di step ini."
                )

    # Cek: text tidak kosong
    if not step.text.strip():
        warnings.append("SimulationStep.text kosong — frontend akan menampilkan pesan kosong.")

    # Cek: hint tanpa node target (OK tapi catat)
    for action in step.actions:
        if action.type == "trigger_hint" and not action.data.target_node_id:
            warnings.append(
                f"Hint untuk checklist '{action.data.checklist_id}' tidak punya target_node_id. "
                "Tidak ada node yang akan di-highlight di whiteboard."
            )

    is_valid = len(warnings) == 0
    if warnings:
        for w in warnings:
            logger.warning(f"[StepValidation] {w}")

    return is_valid, warnings