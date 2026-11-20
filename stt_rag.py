import os
import sys
import argparse
import numpy as np
import whisper
from rag import get_or_build_vectorstore, get_llm, ask_syllabus


def transcribe_from_file(audio_path: str, model_size: str = "base") -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file '{audio_path}' not found.")
    print(f"Loading Whisper '{model_size}' model and transcribing '{audio_path}'...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, fp16=False, language="en")
    return result["text"].strip()


def transcribe_from_microphone(duration: int = 5, sample_rate: int = 16000, model_size: str = "base") -> str:
    import sounddevice as sd

    print(f"Recording from microphone for {duration} seconds... Speak your syllabus question now.")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Recording complete!")

    audio = audio.flatten().astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val

    print(f"Transcribing audio with Whisper '{model_size}'...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio, fp16=False, language="en")
    return result["text"].strip()


def main():
    parser = argparse.ArgumentParser(description="Part 4.1: Speech-to-Text RAG Pipeline")
    parser.add_argument("--audio", type=str, default=None, help="Path to input WAV/MP3 audio file")
    parser.add_argument("--duration", type=int, default=6, help="Microphone recording duration in seconds")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model size (tiny, base, small)")
    args = parser.parse_args()

    print("=" * 70)
    print("PART 4.1: SPEECH-TO-TEXT (STT) + SYLLABUS RAG")
    print("=" * 70)

    if args.audio:
        transcribed_text = transcribe_from_file(args.audio, model_size=args.whisper_model)
    else:
        try:
            transcribed_text = transcribe_from_microphone(
                duration=args.duration, model_size=args.whisper_model
            )
        except Exception as e:
            print(f"\n[Environment Notice] Microphone capture unavailable ({e}).")
            print("Tip: Pass an audio file instead using: python stt_rag.py --audio sample_question.wav")
            sys.exit(1)

    print(f"\nTranscribed Question: {transcribed_text}")

    vectorstore = get_or_build_vectorstore()
    llm = get_llm()
    rag_result = ask_syllabus(transcribed_text, vectorstore=vectorstore, llm=llm)

    print("\nRAG Generated Answer:")
    print(rag_result["answer"])


if __name__ == "__main__":
    main()
