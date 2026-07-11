# ClinicThinking AI - OSCE Simulator Agent (Gemini Orchestrator)

Repositori ini berisi agen kecerdasan buatan (*AI Agent*) yang berfungsi sebagai pasien simulasi dan penguji dalam platform simulator OSCE ClinicThinking. AI ini mengelola jalannya simulasi klinis secara asinkron menggunakan model bahasa **Google Gemini** dan berintegrasi ke ruang WebRTC menggunakan **LiveKit Agents SDK**.

## Cara Kerja Agen AI
1. **LiveKit Voice Integration**: Agen mendengarkan suara mahasiswa dari room LiveKit menggunakan WebRTC, mengonversi audio ke teks (STT), dan merespons kembali menggunakan Text-to-Speech (TTS) natural.
2. **Clinical Reasoning Decision Engine**: Agen membaca berkas *Illness Script* dari database dan berperan sesuai dengan profil pasien medis yang diberikan (termasuk menahan informasi penunjang hingga mahasiswa menanyakannya secara aktif).
3. **Whiteboard Action Generator**: Agen menganalisis alur percakapan secara dinamis dan menghasilkan perintah modifikasi whiteboard (seperti `add_node`, `add_edge`, atau `trigger_hint`) untuk disinkronkan ke layar mahasiswa.

## Teknologi & Library
- **Bahasa**: Python 3.11+
- **LLM Engine**: Google Gemini API (`google-generativeai`)
- **WebRTC & Agent Framework**: LiveKit Agents SDK (`livekit-agents` & `livekit-api`)
- **Protokol Komunikasi**: gRPC (untuk sinkronisasi state whiteboard ke backend)
- **Format Data**: Protocol Buffers (Protobuf)

## Persyaratan Sistem
- Python 3.11 atau lebih baru
- Google Gemini API Key

## Pengembangan Lokal

1. Clone repositori ini:
   ```bash
   git clone https://github.com/adiradmnk/OSCE-Simulation-AI.git
   cd OSCE-Simulation-AI
   ```

2. Buat Virtual Environment dan install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Setup environment variables (`.env`):
   ```env
   LIVEKIT_URL=ws://localhost:7880
   LIVEKIT_API_KEY=devkey
   LIVEKIT_API_SECRET=devsecret
   GOOGLE_API_KEY=Kunci_API_Gemini_Anda
   ```

4. Jalankan agen AI dalam mode development:
   ```bash
   python main.py dev
   ```
   Agen akan otomatis mendengarkan job yang masuk dari server LiveKit lokal.

## Kontribusi Kasus Baru (Casebank)
Struktur skenario klinis dan illness script untuk agen AI didefinisikan secara deklaratif di dalam berkas `prompts/casebank.json`. Anda bisa berkontribusi menambahkan variasi penyakit baru dengan mengikuti skema JSON yang ada di dalamnya.
