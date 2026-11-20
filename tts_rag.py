import os
import sys
import platform
import subprocess
import argparse
from rag import get_or_build_vectorstore, get_llm, ask_syllabus


def synthesize_speech(text: str, output_file: str = "rag_answer.mp3") -> str:
    """
    Converts text to speech using gTTS (or offline pyttsx3 fallback) and saves to output_file.
    """
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en")
        tts.save(output_file)
        print(f"Synthesized speech via gTTS -> '{output_file}'")
        return output_file
    except Exception as gtts_err:
        print(f"gTTS unavailable ({gtts_err}); falling back to offline pyttsx3...")
        import pyttsx3
        wav_file = os.path.splitext(output_file)[0] + ".wav"
        engine = pyttsx3.init()
        engine.save_to_file(text, wav_file)
        engine.runAndWait()
        print(f"Synthesized speech via pyttsx3 -> '{wav_file}'")
        return wav_file


def play_audio_file(file_path: str):
    """
    Attempts to play the generated audio file aloud on macOS, Windows, or Linux.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["afplay", file_path], check=True)
        elif system == "Windows":
            os.startfile(file_path)
        else:
            # Linux: try mpg123, ffplay, or aplay
            for player in (["mpg123", "-q", file_path], ["ffplay", "-nodisp", "-autoexit", file_path], ["aplay", file_path]):
                if subprocess.call(["which", player[0]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                    subprocess.run(player, check=True)
                    return
            raise RuntimeError("No CLI audio player (mpg123/ffplay/aplay) or audio output device found.")
        print("Audio playback completed.")
    except Exception as e:
        print(f"[Audio Playback Notice] Could not play audio aloud in this environment ({e}).")
        print(f"Spoken audio file is saved at '{file_path}' for playback verification.")


def main():
    parser = argparse.ArgumentParser(description="Part 4.2: Syllabus RAG + Text-to-Speech Pipeline")
    parser.add_argument(
        "--question",
        type=str,
        default="What is the grading policy and late work policy for CPSC 254?",
        help="Question to ask the syllabus RAG system",
    )
    parser.add_argument("--output", type=str, default="rag_answer.mp3", help="Output audio file path")
    parser.add_argument("--no-play", action="store_true", help="Skip live speaker playback (save file only)")
    args = parser.parse_args()

    print("=" * 70)
    print("PART 4.2: SYLLABUS RAG + TEXT-TO-SPEECH (TTS)")
    print("=" * 70)
    print(f"Input Question: {args.question}\n")

    vectorstore = get_or_build_vectorstore()
    llm = get_llm()
    rag_result = ask_syllabus(args.question, vectorstore=vectorstore, llm=llm)
    answer_text = rag_result["answer"]

    print(f"RAG Generated Answer:\n{answer_text}\n")

    saved_audio = synthesize_speech(answer_text, output_file=args.output)
    if not args.no_play:
        play_audio_file(saved_audio)


if __name__ == "__main__":
    main()
