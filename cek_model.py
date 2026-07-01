import os
from google import genai

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

print("🔍 Mencari model yang tersedia...")
try:
    for model in client.models.list():
        print(f"✅ Model: {model.name}")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")


# model yang bisa dipakai
"""
✅ Model: models/gemini-2.5-flash
✅ Model: models/gemini-2.5-pro
✅ Model: models/gemini-2.0-flash
✅ Model: models/gemini-2.0-flash-001
✅ Model: models/gemini-2.0-flash-lite-001
✅ Model: models/gemini-2.0-flash-lite
✅ Model: models/gemini-2.5-flash-preview-tts
✅ Model: models/gemini-2.5-pro-preview-tts
✅ Model: models/gemma-4-26b-a4b-it
✅ Model: models/gemma-4-31b-it
✅ Model: models/gemini-flash-latest
✅ Model: models/gemini-flash-lite-latest
✅ Model: models/gemini-pro-latest
✅ Model: models/gemini-2.5-flash-lite
✅ Model: models/gemini-2.5-flash-image
✅ Model: models/gemini-3-pro-preview
✅ Model: models/gemini-3-flash-preview
✅ Model: models/gemini-3.1-pro-preview
✅ Model: models/gemini-3.1-pro-preview-customtools
✅ Model: models/gemini-3.1-flash-lite-preview
✅ Model: models/gemini-3.1-flash-lite
✅ Model: models/gemini-3-pro-image-preview
✅ Model: models/gemini-3-pro-image
✅ Model: models/nano-banana-pro-preview
✅ Model: models/gemini-3.1-flash-image-preview
✅ Model: models/gemini-3.1-flash-image
✅ Model: models/gemini-3.1-flash-lite-image
✅ Model: models/gemini-3.5-flash
✅ Model: models/gemini-omni-flash-preview
✅ Model: models/lyria-3-clip-preview
✅ Model: models/lyria-3-pro-preview
✅ Model: models/gemini-3.1-flash-tts-preview
✅ Model: models/gemini-robotics-er-1.5-preview
✅ Model: models/gemini-robotics-er-1.6-preview
✅ Model: models/gemini-2.5-computer-use-preview-10-2025
✅ Model: models/antigravity-preview-05-2026
✅ Model: models/deep-research-max-preview-04-2026
✅ Model: models/deep-research-preview-04-2026
✅ Model: models/deep-research-pro-preview-12-2025
✅ Model: models/gemini-embedding-001
✅ Model: models/gemini-embedding-2-preview
✅ Model: models/gemini-embedding-2
✅ Model: models/aqa
✅ Model: models/imagen-4.0-generate-001
✅ Model: models/imagen-4.0-ultra-generate-001
✅ Model: models/imagen-4.0-fast-generate-001
✅ Model: models/veo-3.1-generate-preview
✅ Model: models/veo-3.1-fast-generate-preview
✅ Model: models/veo-3.1-lite-generate-preview
✅ Model: models/gemini-2.5-flash-native-audio-latest
✅ Model: models/gemini-2.5-flash-native-audio-preview-09-2025
✅ Model: models/gemini-2.5-flash-native-audio-preview-12-2025
✅ Model: models/gemini-3.1-flash-live-preview
✅ Model: models/gemini-3.5-live-translate-preview
"""