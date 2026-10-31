import speech_recognition as sr

recognizer = sr.Recognizer()
with sr.Microphone() as source:
    print("Please ask a question about the syllabus...")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    audio_data = recognizer.listen(source)
    
    try:
        # Transcribe speech
        question = recognizer.recognize_google(audio_data)
        print(f"Transcription: {question}")
        
        # TODO: Pass 'question' to your RAG pipeline and print the generated answer
        
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
