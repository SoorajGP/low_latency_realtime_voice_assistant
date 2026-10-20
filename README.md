A real-time conversational AI assistant built as part of the XIBOTIX AI Engineer Internship Assignment.

The assistant accepts voice input, processes it completely offline, generates a response using a local LLM, and replies with natural speech.

The assignment looked straightforward at first, but while building it I realized the biggest challenge wasn't speech recognition or the LLM—it was latency. A simple sequential pipeline feels noticeably slow because every stage waits for the previous one to finish.

Instead of optimizing a single component, I focused on making the entire pipeline feel more responsive by streaming responses, running tasks concurrently, and handling delays gracefully.

Features
Offline speech-to-text using faster-whisper
Offline LLM using Ollama + phi3:mini
Local text-to-speech using pyttsx3
Streaming response generation
Sentence-level speech streaming
Automatic fallback responses during slow inference
Modular architecture with separate components for STT, LLM, TTS and orchestration
Pipeline
User Speech - Speech Recognition (faster-whisper) - Local LLM (Ollama) - Streaming Tokens - Sentence Buffer - Text-to-Speech - Assistant Response

Unlike a traditional pipeline, the assistant does not wait for the full response before speaking. As soon as the model finishes a complete sentence, it is sent to the TTS engine while the LLM continues generating the rest of the answer.

This makes conversations feel much more natural, especially on CPU-only hardware.

Why I Chose These Tools
Speech-to-Text — faster-whisper
  I initially considered the standard Whisper implementation, but CPU inference was too slow for a conversational assistant.
  I switched to faster-whisper using the Tiny model with int8 quantization, which offers a good balance between speed and transcription quality.

Language Model — Ollama + Phi-3 Mini
  This assignment preferred an offline solution, so I wanted a model that could run locally without relying on cloud APIs.
  I chose phi3 because it is lightweight enough for my hardware while still producing coherent conversational responses.

Text-to-Speech — pyttsx3
  pyttsx3 isn't the most realistic voice engine available, but it is:
    Offline
    Lightweight
    Easy to integrate
    Fast enough for this prototype

    For a production robotic system, I would replace it with a neural TTS model.

Streaming Responses
  One thing I noticed early was that waiting for the LLM to finish generating an entire paragraph before speaking made the assistant feel slow.
  To avoid that, I added a small SentenceBuffer. Instead of waiting for the complete response, it watches the streaming tokens coming from Ollama. Whenever a complete sentence is detected (., ?, or !), that sentence is immediately sent to the TTS engine. While the first sentence is being spoken, the LLM continues generating the remaining response in the background.

Handling Delays
  The assignment specifically mentioned keeping the user engaged when processing takes longer than expected.
  To handle this, I implemented a small timer. After transcription finishes, an 0.8s timer starts.

If the LLM hasn't produced its first token before the timer expires, the assistant plays a short filler phrase such as:
  "Hmm, let me think about that.",
    "One moment, please.",
    "Working on that for you.",
    "Let me consider that."

Once the first token arrives, the timer is cancelled automatically.

The goal isn't to hide latency completely, but to make the interaction feel less awkward than sitting in silence.

Project Structure
main.py
├── SpeechToText
├── OllamaClient
├── TextToSpeechEngine
├── SentenceBuffer
├── FillerMechanism
└── VoiceAssistant

Each component has a single responsibility, making it easier to extend or replace individual modules.

For example:

swap Whisper with another STT engine
replace Phi-3 with another Ollama model
replace pyttsx3 with XTTS
add conversation memory
integrate wake-word detection

without changing the overall architecture.

Installation
Prerequisites
  Python 3.10+
  Ollama

Install the local model:

  ollama pull phi3:mini
  Setup
  python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt

If PyAudio fails to install on Windows:

  pip install pipwin
  pipwin install pyaudio
  Running
  python main.py

The assistant continuously listens for speech and responds once an utterance is detected.

The following values can be adjusted depending on the hardware and desired user experience.
  Variable	Description
  MIC_PAUSE_THRESHOLD	Silence duration before processing speech
  FILLER_TIMEOUT_SECONDS	Delay before playing a filler response
  OLLAMA_MODEL	Local LLM used for generation
  WHISPER_MODEL_SIZE	Speech recognition model
  SYSTEM_PROMPT	Assistant behaviour
  Limitations

This is a prototype intended for the internship assignment.

Current limitations include:

  English-only speech recognition
  No wake-word detection
  No conversation memory
  CPU performance depends on the host machine
  Basic offline TTS voice
  Future Improvements

If I were extending this for an actual humanoid robot, I would focus on:

  Wake-word detection using Porcupine
  Neural TTS (XTTS v2)
  Long-term conversation memory
  Vision-language integration
  RAG over robot manuals and documentation
  GPU acceleration on edge devices such as NVIDIA Jetson

Thanks for taking the time to review my submission. I enjoyed building this project and, more importantly, understanding the engineering trade-offs involved in making conversational AI feel responsive on local hardware.