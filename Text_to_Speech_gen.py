# TTS is a library for advanced Text-to-Speech generation.

# https://github.com/coqui-ai/TTS


# Example
# Running a multi-speaker and multi-lingual model
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models and choose the first one
model_name = TTS().list_models()[0]
# Init TTS
tts = TTS(model_name).to(device)

# Run TTS
# ❗ Since this model is multi-speaker and multi-lingual, we must set the target speaker and the language
# Text to speech with a numpy output
wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[0], language=tts.languages[0])
# Text to speech to a file
tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")


# Running a single speaker model
# Init TTS with the target model name
tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False).to(device)

# Run TTS
tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path=OUTPUT_PATH)

# Example voice cloning with YourTTS in English, French and Portuguese
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
tts.tts_to_file("This is voice cloning.", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")
tts.tts_to_file("C'est le clonage de la voix.", speaker_wav="my/cloning/audio.wav", language="fr-fr", file_path="output.wav")
tts.tts_to_file("Isso é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt-br", file_path="output.wav")


# Example voice conversion
# Converting the voice in source_wav to the voice of target_wav

tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to("cuda")
tts.voice_conversion_to_file(source_wav="my/source.wav", target_wav="my/target.wav", file_path="output.wav")


# XTTS model
models = TTS(cs_api_model="XTTS").list_models()
# Init TTS with the target studio speaker
tts = TTS(model_name="coqui_studio/en/Torcull Diarmuid/coqui_studio", progress_bar=False)
# Run TTS
tts.tts_to_file(text="This is a test.", file_path=OUTPUT_PATH)


# V1 model
models = TTS(cs_api_model="V1").list_models()
# Run TTS with emotion and speed control
# Emotion control only works with V1 model
tts.tts_to_file(text="This is a test.", file_path=OUTPUT_PATH, emotion="Happy", speed=1.5)


# Example voice cloning together with the voice conversion model.
# This way, you can clone voices by using any model in TTS.

tts = TTS("tts_models/de/thorsten/tacotron2-DDC")
tts.tts_with_vc_to_file(
    "Wie sage ich auf Italienisch, dass ich dich liebe?",
    speaker_wav="target/speaker.wav",
    file_path="output.wav"
)

# XTTS-multilingual

models = TTS(cs_api_model="XTTS-multilingual").list_models()
# Run TTS with emotion and speed control
# Emotion control only works with V1 model
tts.tts_to_file(text="Das ist ein Test.", file_path=OUTPUT_PATH, language="de", speed=1.0)


# TTS with on the fly voice conversion
api = TTS("tts_models/deu/fairseq/vits")
api.tts_with_vc_to_file(
    "Wie sage ich auf Italienisch, dass ich dich liebe?",
    speaker_wav="target/speaker.wav",
    file_path="output.wav"
)
