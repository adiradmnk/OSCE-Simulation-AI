"""
main.py — ClinicThinking AI Agent Entrypoint
Kompatibel dengan livekit-agents==1.3.12

CATATAN PENTING tentang on_data_received:
─────────────────────────────────────────────────────────────────
livekit-agents >= 1.0.x MENGUBAH cara memanggil callback data_received.
SDK memanggil callback dengan SATU argumen (DataPacket object), bukan 3
argumen terpisah seperti versi lama.

    ❌ LAMA (crash di 1.x): def on_data_received(data, participant, kind)
    ✅ BARU (1.x):          def on_data_received(data_packet: rtc.DataPacket)

DataPacket memiliki atribut:
    .data        → bytes — payload raw
    .participant → rtc.RemoteParticipant — pengirim
    .kind        → rtc.DataPacketKind — RELIABLE / LOSSY
    .topic       → Optional[str] — topik (jika diset)
─────────────────────────────────────────────────────────────────
"""

import asyncio
import json
import logging
import os
from livekit.agents import WorkerOptions, cli, JobContext
from livekit import rtc
from agent.director import SimulationDirector as Director

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def entrypoint(ctx: JobContext):
    # ── Initialize Database Connection Pool ─────────────────────────
    db_pool = None
    try:
        import asyncpg
        db_url = os.environ.get("DATABASE_URL", "postgresql://clinicthinking:password@postgres:5432/clinicthinking")
        db_pool = await asyncpg.create_pool(db_url)
        logger.info("Database pool initialized successfully.")
    except Exception as e:
        logger.error(f"Gagal inisialisasi database pool: {e!r}")

    # ── 1. Ambil metadata room (case_id, dll.) ──────────────────────
    metadata: dict = {}
    try:
        if ctx.room.metadata:
            metadata = json.loads(ctx.room.metadata)
    except Exception as e:
        logger.error(f"Gagal parse room metadata: {e}")

    case_id = metadata.get("case_id", "RESP-001")
    logger.info(f"Room {ctx.room.name!r} → kasus: {case_id!r}")

    # ── 2. Inisialisasi Director ────────────────────────────────────
    director = Director(case_id=case_id, session_id=ctx.room.name, db_pool=db_pool)

    # ── Initialize AudioSource and LocalAudioTrack ──
    # Gemini native audio menghasilkan raw PCM 24kHz mono (1 channel, 16-bit)
    sample_rate = 24000
    num_channels = 1
    audio_source = rtc.AudioSource(sample_rate, num_channels)
    audio_track = rtc.LocalAudioTrack.create_audio_track("agent_voice", audio_source)

    # Lock untuk memastikan audio diputar secara berurutan dan tidak tumpang tindih
    audio_lock = asyncio.Lock()

    # Welcome message trigger logic (menghindari race condition)
    welcome_sent = False

    async def trigger_welcome():
        nonlocal welcome_sent
        if welcome_sent:
            return
        
        # Double check database to prevent duplicates on refresh
        if db_pool:
            try:
                count = await db_pool.fetchval(
                    "SELECT COUNT(*) FROM session_events WHERE session_id = $1",
                    ctx.room.name
                )
                if count > 0:
                    logger.info(f"[WelcomeCheck] Sesi sudah memiliki {count} events di DB. Membatalkan trigger_welcome.")
                    welcome_sent = True
                    return
            except Exception as e:
                logger.error(f"[WelcomeCheck] Gagal verifikasi DB di trigger_welcome: {e!r}")

        welcome_sent = True
        logger.info("Mengirim sesi pembuka (welcome message)...")
        initial_step = director.start_session()
        await publish_ai_response(ctx, initial_step, director, audio_source, audio_lock)

    async def trigger_welcome_after_delay():
        await asyncio.sleep(1.5)  # Beri waktu 1.5 detik agar client siap menerima data channel
        await trigger_welcome()

    # ── 3. Daftarkan listener data_received SEBELUM connect ─────────
    @ctx.room.on("data_received")
    def on_data_received(data_packet: rtc.DataPacket):
        # ── Ekstraksi aman dari DataPacket ─────────────────────────
        try:
            raw_bytes: bytes = bytes(data_packet.data)
            participant: rtc.RemoteParticipant | None = getattr(data_packet, "participant", None)
            kind: rtc.DataPacketKind = getattr(data_packet, "kind", rtc.DataPacketKind.KIND_RELIABLE)
        except (AttributeError, TypeError) as e:
            logger.error(f"Gagal ekstrak DataPacket: {e!r} — data_packet={data_packet!r}")
            return

        sender_id = getattr(participant, "identity", "unknown")
        kind_name = "RELIABLE" if kind == rtc.DataPacketKind.KIND_RELIABLE else "LOSSY"
        logger.info(f"Data diterima dari '{sender_id}' [{kind_name}] — {len(raw_bytes)} bytes")

        # ── Parse JSON ─────────────────────────────────────────────
        try:
            payload: dict = json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Payload bukan JSON valid: {e!r} — raw={raw_bytes[:200]!r}")
            return

        msg_type: str = payload.get("type", "")
        logger.info(f"Tipe pesan: {msg_type!r}")

        # ── Routing pesan ──────────────────────────────────────────
        try:
            if msg_type == "user_input":
                content = payload.get("content", "").strip()
                if content:
                    asyncio.create_task(
                        process_chat(ctx, director, content, audio_source, audio_lock),
                        name=f"process_chat_{ctx.room.name}"
                    )
                else:
                    logger.warning("user_input diterima tapi content kosong, diabaikan.")

            elif msg_type == "client_ready":
                logger.info("client_ready diterima dari frontend.")
                asyncio.create_task(trigger_welcome())

            else:
                logger.debug(f"Tipe tidak ditangani, diabaikan: {msg_type!r}")

        except Exception as e:
            logger.error(f"Error saat routing pesan '{msg_type}': {e!r}", exc_info=True)

    # ── 4. Connect ke LiveKit room ──────────────────────────────────
    await ctx.connect()
    logger.info(f"Agent terhubung ke room: {ctx.room.name!r}")

    # ── Publish audio track ──
    try:
        publication = await ctx.room.local_participant.publish_track(
            audio_track,
            rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_MICROPHONE,
                name="agent_voice"
            )
        )
        logger.info(f"Audio track successfully published: {publication.sid}")
    except Exception as e:
        logger.error(f"Gagal publish audio track: {e!r}", exc_info=True)

    # ── 4b. Ambil room metadata setelah koneksi sukses (untuk kepastian data terbaru)
    metadata: dict = {}
    try:
        raw_meta = ctx.room.metadata
        logger.info(f"Metadata dari LiveKit room raw: {raw_meta!r}")
        if raw_meta:
            metadata = json.loads(raw_meta)
    except Exception as e:
        logger.error(f"Gagal parse room metadata setelah connect: {e!r}")

    case_id = metadata.get("case_id", "RESP-001")
    logger.info(f"Menyetel kasus aktif: {case_id!r}")
    director.case_id = case_id
    director._case_data = director._load_case(case_id)

    # ── 4c. Deteksi Sesi Baru secara Proaktif & Aman ─────────────────
    is_new = True
    if db_pool:
        try:
            count = await db_pool.fetchval(
                "SELECT COUNT(*) FROM session_events WHERE session_id = $1",
                ctx.room.name
            )
            is_new = (count == 0)
            logger.info(f"[WelcomeCheck] Event count di DB: {count} -> is_new={is_new}")
        except Exception as e:
            logger.error(f"[WelcomeCheck] Gagal check session_events: {e!r}")

    if is_new:
        if ctx.room.remote_participants:
            logger.info("[WelcomeCheck] User terdeteksi sudah ada di room. Menjadwalkan welcome message...")
            asyncio.create_task(trigger_welcome_after_delay())
        else:
            logger.info("[WelcomeCheck] Belum ada user di room. Menunggu user terhubung...")

        # Daftarkan listener jika user terhubung kemudian
        @ctx.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logger.info(f"[WelcomeCheck] User terhubung: {participant.identity}. Menjadwalkan welcome message...")
            asyncio.create_task(trigger_welcome_after_delay())

    # ── 5. Tahan proses hingga session selesai ──────────────────────
    try:
        await asyncio.Event().wait()
    finally:
        if db_pool:
            await db_pool.close()
            logger.info("Database pool closed.")


async def play_audio_pcm(audio_source: rtc.AudioSource, pcm_bytes: bytes, sample_rate: int = 24000, lock: asyncio.Lock = None) -> None:
    """
    Mengirimkan raw PCM 16-bit ke AudioSource dalam potongan frame 20ms (960 bytes).
    Lock menjamin audio diputar berurutan dan tidak tumpang tindih.
    """
    if not audio_source:
        return

    # Hitung ukuran frame 20ms
    chunk_duration = 0.02  # 20ms
    samples_per_chunk = int(sample_rate * chunk_duration)  # 480 samples
    bytes_per_sample = 2  # 16-bit
    chunk_size = samples_per_chunk * bytes_per_sample  # 960 bytes

    async def _stream():
        logger.info(f"[AudioPlayer] Mulai streaming {len(pcm_bytes)} bytes PCM ({sample_rate}Hz mono) ke AudioSource")
        try:
            for i in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    # Pad dengan silence (zeros) jika di akhir
                    chunk = chunk + b'\x00' * (chunk_size - len(chunk))

                frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=samples_per_chunk
                )
                await audio_source.capture_frame(frame)
                await asyncio.sleep(chunk_duration)
            logger.info("[AudioPlayer] Selesai streaming PCM.")
        except Exception as e:
            logger.error(f"[AudioPlayer] Error streaming PCM: {e!r}")

    if lock:
        async with lock:
            await _stream()
    else:
        await _stream()


async def process_chat(ctx: JobContext, director: Director, user_text: str, audio_source: rtc.AudioSource, audio_lock: asyncio.Lock) -> None:
    """
    Proses input teks user melalui Director AI dan publikasikan hasilnya.
    """
    logger.info(f"Memproses: {user_text[:80]!r}")
    try:
        step = await director.process_turn(user_text)
        await publish_ai_response(ctx, step, director, audio_source, audio_lock)
    except Exception as e:
        logger.error(f"Error process_turn: {e!r}", exc_info=True)
        # Kirim fallback ke frontend agar tidak terlihat diam
        await send_livekit(ctx, {
            "type": "chat_message",
            "role": "ai",
            "content": "Maaf, ada gangguan sesaat. Silakan ulangi pertanyaan Anda.",
            "audio_duration_ms": 0
        })


# Helper untuk mencatat event langsung ke database via Python pgpool
async def log_event_to_db(director, session_id: str, event_type: str, event_data: dict) -> None:
    if not director or not hasattr(director, 'db') or not director.db:
        return
    try:
        # Cari sequence number berikutnya
        query_seq = "SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM session_events WHERE session_id = $1"
        seq_num = await director.db.fetchval(query_seq, session_id)
        
        # Insert event
        query_insert = """
            INSERT INTO session_events (session_id, event_type, event_data, sequence_number)
            VALUES ($1, $2, $3::jsonb, $4)
        """
        await director.db.execute(query_insert, session_id, event_type, json.dumps(event_data), seq_num)
        logger.info(f"[DB-Logger] Sukses menyimpan event '{event_type}' ke database (seq: {seq_num})")
    except Exception as e:
        logger.error(f"[DB-Logger] Gagal menyimpan event '{event_type}': {e!r}")


async def publish_ai_response(ctx: JobContext, step, director=None, audio_source: rtc.AudioSource = None, audio_lock: asyncio.Lock = None) -> None:
    """
    Kirim SimulationStep ke frontend via LiveKit Data Channel (RELIABLE)
    dan catat secara persisten ke database PostgreSQL.
    """
    if step is None:
        logger.warning("publish_ai_response dipanggil dengan step=None")
        return

    session_id = ctx.room.name
    narrative = (step.text or "").strip()

    audio_data = b""
    duration_ms = 0

    # 1. Generate audio terdefinisi untuk memperoleh durasi suara
    if narrative:
        audio_data = await director.generate_tts_audio(narrative)
        if audio_data:
            # 24000Hz 16-bit mono: 48000 bytes per second
            duration_ms = int((len(audio_data) / 48000.0) * 1000)

    # 2. Kirim pesan chat dengan durasi suara ke frontend
    if narrative:
        await send_livekit(ctx, {
            "type": "chat_message",
            "role": "ai",
            "content": narrative,
            "audio_duration_ms": duration_ms
        })
        logger.info(f"chat_message dikirim: {narrative[:80]!r} (durasi: {duration_ms}ms)")
        
        # Simpan ke DB secara langsung
        if director:
            await log_event_to_db(director, session_id, 'ai_response', {
                "type": "chat_message",
                "role": "ai",
                "content": narrative,
                "audio_duration_ms": duration_ms
            })

    # 3. Kirim whiteboard actions
    for action in (step.actions or []):
        try:
            action_dict = action.model_dump()
            if action_dict.get("type") == "no_action":
                continue

            await send_livekit(ctx, {
                "type": "ai_action",
                "payload": action_dict,
            })
            logger.info(f"ai_action: payload.type={action_dict.get('type')!r}")
            
            # Simpan ke DB secara langsung
            if director:
                await log_event_to_db(director, session_id, 'ai_action', {
                    "type": "ai_action",
                    "payload": action_dict
                })
        except Exception as e:
            logger.error(f"Gagal kirim/simpan ai_action: {e!r}", exc_info=True)

    # 4. Stream audio via LiveKit AudioTrack
    if audio_data and audio_source:
        asyncio.create_task(
            play_audio_pcm(audio_source, audio_data, sample_rate=24000, lock=audio_lock),
            name=f"play_audio_{ctx.room.name}"
        )

    # 5. Timer control
    if step.timer_action:
        timer_val = (
            step.timer_action.value
            if hasattr(step.timer_action, "value")
            else str(step.timer_action)
        )
        await send_livekit(ctx, {"type": "timer_control", "action": timer_val})
        logger.info(f"timer_control: {timer_val!r}")


# ─── Send Helper ─────────────────────────────────────────────────────

async def send_livekit(ctx: JobContext, payload: dict) -> None:
    """
    Encode payload dict → UTF-8 JSON bytes dan publish ke room dengan
    DataPacketKind.RELIABLE agar pesan tidak hilang.
    """
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await ctx.room.local_participant.publish_data(
            data,
            reliable=True,   # Menggunakan DataPacketKind.RELIABLE
        )
        logger.debug(f"publish_data OK — type={payload.get('type')!r} ({len(data)} bytes)")
    except Exception as e:
        logger.error(f"Gagal publish_data (type={payload.get('type')!r}): {e!r}")


# ─── Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))