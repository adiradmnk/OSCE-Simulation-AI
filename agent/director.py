from __future__ import annotations

import json
import logging
import os
import time
import re
import asyncio
from pathlib import Path
from typing import Any
from google import genai
from google.genai import types

from openai import AsyncOpenAI, OpenAI


from models.schemas import (
    BiasAnalysis,
    BiasType,
    SessionContext,
    SimulationStep,
    TimerAction,
    NoAction,
)
from utils.response_parser import (
    parse_llm_response,
    parse_and_inject_bias,
    validate_step,
)

logger = logging.getLogger(__name__)


CASE_BANK_PATH   = Path(__file__).parent.parent / "prompts" / "casebank.json"
PROMPT_PATH      = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"
DEFAULT_MODEL    = "gpt-4o-mini"

BRAIN_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-3.1-flash-lite",
    "models/gemini-3.5-flash"
]

AUDIO_MODELS = [
    "models/gemini-2.5-flash-preview-tts",
    "models/gemini-2.5-pro-preview-tts",
    "models/gemini-3.1-flash-tts-preview"
]

MAX_HISTORY_TURNS = 12      
SESSION_DURATION  = 600        


PREMATURE_CLOSURE_MIN_CHECKLIST_PCT = 0.30  
PREMATURE_CLOSURE_MAX_TURN          = 6      
ANCHORING_SAME_HYP_TURNS            = 3     


DIAGNOSIS_KEYWORDS = [
    "diagnosis", "diagnosisnya", "diagnosa", "saya yakin", "sudah pasti",
    "pastinya", "kemungkinan besar", "kesimpulan", "final diagnosis",
    "langsung terapi", "berikan obat", "resepkan", "tidak perlu tanya lagi",
    "sudah cukup", "saya rasa ini", "ini adalah", "pasien ini menderita",
]




class SimulationDirector:
    def __init__(
        self,
        case_id: str,  
        session_id: str,
        model:      str = DEFAULT_MODEL,
        use_async:  bool = True,
        db_pool=None
    ):
        self.case_id    = case_id
        self.session_id = session_id
        self.model      = model
        self.use_async  = use_async
        self.db = db_pool

        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.brain_model = 'gemini-3.1-flash-lite'
        self.tts_model   = "gemini-3.1-flash-tts-preview"

        # Load case bank & system prompt template
        self._case_data    = self._load_case(case_id)
        self._prompt_tmpl  = PROMPT_PATH.read_text(encoding="utf-8")

        # Session state
        self._context      = SessionContext(case_id=case_id, turn_number=0)
        self._history: list[dict[str, str]] = []   # OpenAI message format
        self._session_start = time.time()

        # Bias tracking state
        self._last_hypothesis_mentioned: str | None = None
        self._consec_same_hyp_count:     int = 0
        self._biases_this_session:       list[BiasType] = []

        logger.info(
            f"SimulationDirector initialized | "
            f"case={case_id} | session={session_id} | model={model}"
        )

    def load_case_context(self, session_id: str, case_data: dict):
        """Method ini dipanggil oleh main.py saat ada request gRPC."""
        self.session_id = session_id
        # Update data kasus berdasarkan request dari Go
        if "title" in case_data:
            self._case_data["title"] = case_data["title"]
        # Tambahkan logika lain sesuai kebutuhan
        logger.info(f"Context updated for session {session_id}")


    async def process_turn(self, student_input: str) -> SimulationStep:
        logger.info(f"[Director] Memulai process_turn untuk input: {student_input[:60]!r}")
        try:
            self._context.turn_number += 1
            
            # 1. Panggil data bias hasil backend Go
            logger.info(f"[Director] Turn {self._context.turn_number} - Menganalisis bias...")
            bias_info = await self._analyze_bias(self.session_id)
            logger.info(f"[Director] Bias info diperoleh: {bias_info}")

            # 2. Update context
            logger.info("[Director] Memperbarui konteks dari input...")
            self._update_context_from_input(student_input)
            
            # 3. Panggil LLM
            logger.info("[Director] Memanggil LLM (Gemini)...")
            messages = self._build_messages(student_input)
            logger.info(f"[Director] Prompt Messages: {json.dumps(messages, ensure_ascii=False)[:300]}...")
            raw_response = await self._call_llm_async(messages)
            logger.info(f"[Director] Mentah respons LLM (pertama 200 karakter): {raw_response[:200]!r}")

            # 4. Finalisasi: Inject bias jika bias_info ada
            logger.info("[Director] Memulai parsing respons LLM...")
            if bias_info:
                step = parse_and_inject_bias(
                    raw=raw_response,
                    bias_type=bias_info['type'],
                    bias_description=bias_info['note']
                )
            else:
                step = parse_llm_response(raw_response)
            logger.info(f"[Director] Respons berhasil diparse ke SimulationStep. Text: {step.text[:60]!r}, Actions: {len(step.actions)}")

            # 5. Lanjut sisa proses dan simpan ke history percakapan
            self._update_context_from_step(step)
            self._append_to_history(student_input, step)
            logger.info("[Director] Selesai process_turn dengan sukses.")
            return step
        except Exception as e:
            logger.error(f"[Director] GAGAL di dalam process_turn: {e!r}", exc_info=True)
            raise e

    def process_turn_sync(self, student_input: str) -> SimulationStep:
        """Synchronous version untuk testing atau non-async context."""
        import asyncio
        return asyncio.run(self.process_turn(student_input))

    def start_session(self) -> SimulationStep:
        """
        Kembalikan SimulationStep pembuka sesi.
        Dipanggil saat mahasiswa pertama kali connect ke WebSocket.
        Menyertakan node awal 'Keluhan Utama' di whiteboard.
        """
        from models.schemas import AddNodeAction, AddNodeData, NodeType
        import re

        profile = self._case_data["patient_presentation"]
        name    = profile.get("name_placeholder", "Pasien")
        age     = profile.get("age", "dewasa")
        gender  = profile.get("gender", "")
        chief   = profile.get("chief_complaint", "keluhan tidak disebutkan")

        opening = (
            f"Selamat datang di simulasi OSCE. "
            f"Pasien Anda adalah {name}, {age} tahun, {gender}. "
            f"Keluhan utama: {chief}. "
            f"Waktu Anda adalah 10 menit. Silakan mulai anamnesis."
        )

        # Bersihkan keluhan utama untuk label node keluhan utama yang rapi
        chief_clean = chief.split(",")[0].split(";")[0]
        for word in [" sejak", " yang", " saat", " setelah", " terutama"]:
            if word in chief_clean.lower():
                chief_clean = chief_clean.lower().split(word)[0].strip()
        chief_label = f"Keluhan Utama: {chief_clean.title()}"
        if len(chief_label) > 40:
            chief_label = chief_label[:37] + "..."

        chief_node_id = "sym_" + re.sub(r"[^a-z0-9]", "_", chief.lower())[:30].strip("_")

        initial_node = AddNodeAction(
            type="add_node",
            data=AddNodeData(
                id=chief_node_id,
                label=chief_label,
                type=NodeType.SYMPTOM,
                checklist_ref=None,
            )
        )

        # Update context agar node ini masuk ke canvas tracking
        self._context.node_ids_on_canvas.append(chief_node_id)

        return SimulationStep(
            text=opening,
            actions=[initial_node],
            timerAction=TimerAction.START,
        )

    def end_session(self) -> SimulationStep:
        """SimulationStep penutup sesi — dipanggil saat timer habis."""
        completed  = len(self._context.completed_checklist)
        total      = 16
        bias_count = len(self._biases_this_session)

        summary = (
            f"Waktu habis. Anda telah mengumpulkan {completed}/{total} item pemeriksaan. "
        )
        if self._biases_this_session:
            bias_labels = ", ".join(b.value.replace("_", " ") for b in self._biases_this_session)
            summary += f"Bias kognitif terdeteksi: {bias_labels}. "
        summary += "Silakan tetapkan diagnosis akhir Anda."

        return SimulationStep(
            text=summary,
            actions=[NoAction(type="no_action", data=None)],
            timerAction=TimerAction.STOP,
        )

    def update_checklist(self, checklist_id: str) -> None:
        """Dipanggil oleh handler saat mahasiswa menyelesaikan item checklist."""
        if checklist_id not in self._context.completed_checklist:
            self._context.completed_checklist.append(checklist_id)

    def update_canvas_nodes(self, node_ids: list[str]) -> None:
        """Sync node IDs dari frontend ke context (dipanggil setelah apply action)."""
        self._context.node_ids_on_canvas = node_ids

    async def _analyze_bias(self, session_id: str):
        """
        Mengambil hasil deteksi bias yang sudah diproses backend Go dari DB.
        Mengembalikan None jika db_pool tidak disediakan atau tidak ada data.
        """
        # Guard: jika tidak ada koneksi DB, skip saja
        if self.db is None:
            return None

        query = """
            SELECT bias_type, detected_at_sequence, confidence_note 
            FROM bias_detections 
            WHERE session_id = $1 
            ORDER BY detected_at_sequence DESC 
            LIMIT 1
        """
        try:
            bias_row = await self.db.fetchrow(query, session_id)
            if bias_row:
                return {
                    "type": bias_row['bias_type'],
                    "note": bias_row['confidence_note']
                }
        except Exception as e:
            logger.warning(f"[_analyze_bias] Gagal query DB: {e}")
        return None

    def _extract_hypothesis_mention(self, text: str) -> str | None:
        """
        Ekstrak nama hipotesis yang disebutkan dalam input mahasiswa.
        Dibandingkan dengan differential diagnoses dari case bank.
        """
        differentials = self._case_data.get("differential_diagnoses", [])
        primary = self._case_data["illness_script"]["primary_diagnosis"].lower()

        candidates = [primary] + [
            d["diagnosis"].lower() for d in differentials
        ]

        for cand in candidates:
            # Match whole word, case insensitive
            if re.search(r'\b' + re.escape(cand.split()[0]) + r'\b', text):
                return cand.title()

        # Common abbreviations
        abbrev_map = {
            r'\btb\b':        "Tuberkulosis Paru",
            r'\btbc\b':       "Tuberkulosis Paru",
            r'\bpneumonia\b': "Pneumonia Komunitas",
            r'\bppok\b':      "Bronkitis Kronik",
            r'\bca paru\b':   "Karsinoma Paru",
            r'\bkanker paru\b': "Karsinoma Paru",
        }
        for pattern, name in abbrev_map.items():
            if re.search(pattern, text):
                return name

        return None

    # ── Prompt Building ──────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """
        Isi template system_prompt.txt dengan data kasus dan konteks sesi saat ini.
        """
        case    = self._case_data
        script  = case["illness_script"]
        profile = case["patient_presentation"]
        ctx     = self._context

        # Format checklist
        checklist = case.get("osce_checklist", {})
        checklist_text = "\n".join([
            f"  Anamnesis: {', '.join(checklist.get('anamnesis_items', []))}",
            f"  Physical Exam: {', '.join(checklist.get('physical_exam_items', []))}",
            f"  Workup: {', '.join(checklist.get('expected_workup', []))}",
        ])

        # Format differentials
        diff_text = "\n".join(
            f"  - {d['diagnosis']}: {d['distinguishing_features']}"
            for d in case.get("differential_diagnoses", [])
        )

        # Format consequences
        conseq = script.get("consequences", {})
        conseq_text = ", ".join(
            conseq.get("symptoms", []) +
            conseq.get("physical_signs", [])
        )

        bias_triggers = case.get("common_bias_triggers", {})

        return self._prompt_tmpl.format(
            case_title               = case.get("title", "OSCE Case"),
            primary_diagnosis        = script.get("primary_diagnosis", ""),
            patient_profile          = json.dumps(profile, ensure_ascii=False),
            enabling_conditions      = ", ".join(script.get("enabling_conditions", [])),
            fault_pathophysiology    = script.get("fault_pathophysiology", ""),
            consequences             = conseq_text,
            differential_diagnoses   = diff_text,
            osce_checklist           = checklist_text,
            premature_closure_risk   = bias_triggers.get("premature_closure_risk", ""),
            anchoring_risk           = bias_triggers.get("anchoring_risk", ""),
            turn_number              = ctx.turn_number,
            elapsed_seconds          = ctx.elapsed_seconds,
            completed_checklist      = json.dumps(ctx.completed_checklist, ensure_ascii=False),
            node_ids_on_canvas       = json.dumps(ctx.node_ids_on_canvas, ensure_ascii=False),
            student_hypothesis       = ctx.student_hypothesis or "Belum disebutkan",
            detected_biases          = json.dumps(
                [b.value for b in ctx.detected_biases], ensure_ascii=False
            ),
        )

    def _build_messages(self, student_input: str) -> list[dict[str, str]]:
        """
        Build messages array untuk OpenAI API dengan sliding window history.
        """
        system_prompt = self._build_system_prompt()

        # Sliding window: ambil MAX_HISTORY_TURNS terakhir
        recent_history = self._history[-(MAX_HISTORY_TURNS * 2):]

        messages = [
            {"role": "system", "content": system_prompt},
            *recent_history,
            {"role": "user", "content": student_input},
        ]

        return messages
    
    async def _get_bias_from_db(self, session_id: str):
        if self.db is None:
            return None
        query = """
            SELECT bias_type, confidence_note 
            FROM bias_detections 
            WHERE session_id = $1 
            ORDER BY detected_at_sequence DESC 
            LIMIT 1
        """
        row = await self.db.fetchrow(query, session_id)
        return row 


    # ── LLM Calls ────────────────────────────────────────────────────

    async def _call_llm_async(self, messages: list[dict[str, str]]) -> str:
        contents = []
        for msg in messages:
            if msg["role"] == "system": continue
            # Memastikan format content sesuai dengan SDK google-genai
            contents.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}]
            })

        # Ambil system prompt yang dinamis
        system_instruction_text = [m["content"] for m in messages if m["role"] == "system"][0]

        # Coba memanggil model dari BRAIN_MODELS secara berurutan
        last_error = None
        for model_name in BRAIN_MODELS:
            logger.info(f"[LLM-Fallback] Mencoba model brain: {model_name}")
            try:
                response = await self.client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction_text,
                        response_mime_type="application/json",
                        temperature=0.35,
                    )
                )
                logger.info(f"[LLM-Fallback] Sukses menggunakan model brain: {model_name}")
                return response.text
            except Exception as e:
                logger.warning(f"[LLM-Fallback] Model brain {model_name} gagal: {e!r}")
                last_error = e
                continue

        logger.error(f"[LLM-Fallback] Semua model brain gagal! Error terakhir: {last_error!r}")
        return '{"text": "Maaf, sistem sedang sibuk. Bisa diulangi dok?"}'


    def _call_llm_sync(self, messages: list[dict[str, str]]) -> str:
        """Sync version untuk testing."""
        response = self._client.chat.completions.create(
            model           = self.model,
            messages        = messages,
            response_format = {"type": "json_object"},
            temperature     = 0.35,
            max_tokens      = 800,
            timeout         = 25.0,
        )
        return response.choices[0].message.content or ""

    # ── Parse & Finalize ─────────────────────────────────────────────

    def _parse_and_finalize(self, raw: str, bias: BiasAnalysis) -> SimulationStep:
        # Logic: Jika backend mendeteksi bias, kita memanggil `parse_and_inject_bias`
        # Ini memastikan bias masuk ke dalam action list SEBELUM sampai ke frontend.
        if bias.has_premature_closure or bias.has_anchoring:
            # Panggil fungsi injection dari response_parser
            return parse_and_inject_bias(
                raw=raw,
                bias_type=BiasType.PREMATURE_CLOSURE.value if bias.has_premature_closure else BiasType.ANCHORING_BIAS.value,
                bias_description=bias.closure_reason or bias.anchoring_reason or "",
                affected_hypothesis=self._context.student_hypothesis
            )
        return parse_llm_response(raw)
    

    async def generate_tts_audio(self, text: str) -> bytes:
        """
        Menghasilkan audio TTS dari teks menggunakan model Gemini Audio.
        Mendukung fallback otomatis antar model dalam AUDIO_MODELS.
        """
        # Deteksi peran (Tutor vs Pasien) untuk membedakan suara & tempo
        text_lower = text.lower()
        is_tutor = (
            "selamat datang" in text_lower or 
            "simulasi osce" in text_lower or 
            "waktu anda" in text_lower or 
            "sesi berakhir" in text_lower or
            "mengalihkan" in text_lower
        )
        
        gemini_voice = "Aoede" if is_tutor else "Puck"
        gcp_voice = "id-ID-Standard-C" if is_tutor else "id-ID-Standard-B"
        speaking_rate = 1.15 if is_tutor else 1.25

        logger.info(f"[TTS-Config] Teks terdeteksi sebagai {'Tutor' if is_tutor else 'Pasien'}. Gemini Voice: {gemini_voice}, GCP Voice: {gcp_voice}, Rate: {speaking_rate}")

        # Coba memanggil model dari AUDIO_MODELS secara berurutan
        last_error = None
        for model_name in AUDIO_MODELS:
            logger.info(f"[TTS-Fallback] Mencoba model audio/TTS: {model_name}")
            try:
                config = types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config={
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": gemini_voice
                            }
                        }
                    },
                    temperature=0.7,
                )
                response = await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=model_name,
                        contents=text,
                        config=config
                    ),
                    timeout=4.0
                )
                
                audio_bytes = None
                if hasattr(response, "audio_bytes") and response.audio_bytes:
                    audio_bytes = response.audio_bytes
                else:
                    try:
                        audio_bytes = response.candidates[0].content.parts[0].inline_data.data
                    except Exception:
                        pass
                
                if audio_bytes:
                    logger.info(f"[TTS-Fallback] Sukses menggunakan model audio: {model_name} ({len(audio_bytes)} bytes)")
                    return audio_bytes
                else:
                    logger.warning(f"[TTS-Fallback] Respons model {model_name} tidak mengandung audio bytes.")
            except Exception as e:
                logger.warning(f"[TTS-Fallback] Model audio {model_name} gagal: {e!r}")
                last_error = e
                continue

        # Fallback terakhir: Google Cloud Text-to-Speech API (LINEAR16 24kHz mono)
        logger.info("[TTS-Fallback] Mencoba Google Cloud Text-to-Speech API...")
        try:
            import aiohttp
            import base64
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key:
                url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
                payload = {
                    "input": {"text": text},
                    "voice": {"languageCode": "id-ID", "name": gcp_voice},
                    "audioConfig": {
                        "audioEncoding": "LINEAR16",
                        "sampleRateHertz": 24000,
                        "speakingRate": speaking_rate
                    }
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as resp:
                        if resp.status == 200:
                            res_json = await resp.json()
                            audio_b64 = res_json.get("audioContent", "")
                            if audio_b64:
                                raw_wav = base64.b64decode(audio_b64)
                                # Lewati header WAV (44 bytes) untuk mendapatkan PCM mentah
                                if raw_wav.startswith(b"RIFF"):
                                    pcm_data = raw_wav[44:]
                                else:
                                    pcm_data = raw_wav
                                logger.info(f"[TTS-Fallback] Sukses menggunakan Google Cloud TTS ({len(pcm_data)} bytes PCM)")
                                return pcm_data
                        else:
                            err_text = await resp.text()
                            logger.warning(f"[TTS-Fallback] Google Cloud TTS gagal dengan status {resp.status}: {err_text}")
            else:
                logger.warning("[TTS-Fallback] Google Cloud TTS dibatalkan: API key tidak ditemukan.")
        except Exception as ex:
            logger.warning(f"[TTS-Fallback] Error saat memanggil Google Cloud TTS: {ex!r}")

        logger.error(f"[TTS-Fallback] Semua model audio/TTS gagal! Error terakhir: {last_error!r}")
        return b""



    def _update_context_from_input(self, student_input: str) -> None:
        """Update context berdasarkan analisis input mahasiswa."""
        hyp = self._extract_hypothesis_mention(student_input.lower())
        if hyp:
            self._context.student_hypothesis = hyp

    def _update_context_from_step(self, step: SimulationStep) -> None:
        """
        Update context berdasarkan actions yang dikembalikan LLM.
        Sync node IDs yang baru ditambahkan ke canvas context.
        """
        for action in step.actions:
            if action.type == "add_node":
                node_id = action.data.id
                if node_id not in self._context.node_ids_on_canvas:
                    self._context.node_ids_on_canvas.append(node_id)

                # Jika node punya checklist_ref, update completed checklist
                if action.data.checklist_ref:
                    self.update_checklist(action.data.checklist_ref)

    def _append_to_history(self, student_input: str, step: SimulationStep) -> None:
        self._history.append({"role": "user",      "content": student_input})
        self._history.append({"role": "assistant", "content": step.text})


        max_msgs = MAX_HISTORY_TURNS * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]


    def _load_case(self, case_id: str) -> dict[str, Any]:
        with open(CASE_BANK_PATH, encoding="utf-8") as f:
            data = json.load(f) 


        for system_entry in data.get("systems", []):
            for case in system_entry.get("cases", []):
                if case.get("case_id") == case_id:
                    return case
        
        raise ValueError(f"Case '{case_id}' tidak ditemukan di case_bank.json.")



    @property
    def context(self) -> SessionContext:
        return self._context

    @property
    def bias_history(self) -> list[BiasType]:
        return self._biases_this_session.copy()

    @property
    def case_title(self) -> str:
        return self._case_data.get("title", self.case_id)



 