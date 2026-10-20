"""
Real-Time Offline Voice Assistant

A fully offline, CPU-only, real-time audio-in/audio-out conversational AI
assistant with sub-2-second end-to-end latency.

Pipeline:
    Microphone -> STT (faster-whisper tiny/int8) -> LLM (Ollama local)
               -> Sentence Chunking -> TTS (pyttsx3)
"""

import io
import os
import sys
import json
import time
import wave
import random
import signal
import logging
import tempfile
import threading

from pathlib import Path
from queue import Queue, Empty

import requests
import pyttsx3
import speech_recognition as sr

from faster_whisper import WhisperModel


# Ollama endpoint and model selection.
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"  

# STT configuration -- "tiny" model with int8 quantization for CPU speed.
WHISPER_MODEL_SIZE = "tiny"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# Filler fallback timeout.  If the LLM does not yield its first token within this window, a filler phrase is played to keep the user engaged.
FILLER_TIMEOUT_SECONDS = 2.0

# Directory containing optional pre-recorded .wav filler files
FILLER_AUDIO_DIR = Path(__file__).parent / "filler_audio"

# Filler phrases
FILLER_PHRASES = [
    "Hmm, let me think about that.",
    "One moment, please.",
    "Working on that for you.",
    "Let me consider that.",
]

SYSTEM_PROMPT = (
    "You are a concise, helpful voice assistant. Keep answers short and "
    "conversational -- ideally one to three sentences. Do not use markdown, "
    "bullet points, or numbered lists.  Speak naturally as if in a phone call."
)


MIC_ENERGY_THRESHOLD = 300

MIC_PAUSE_THRESHOLD = 0.8

# Logging configuration.
LOG_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
LOG_LEVEL = logging.INFO

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL, stream=sys.stderr)
logger = logging.getLogger("voice_assistant")



# SpeechToText - faster-whisper wrapper


class SpeechToText:
    """Wraps the faster-whisper model for low-latency CPU transcription.

    The model is loaded once during initialization and reused for every
    transcription call.  Audio data from the SpeechRecognition library is
    written to a temporary WAV file because faster-whisper expects a file
    path or file-like object.
    """

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        logger.info(
            "Loading Whisper model '%s' (device=%s, compute_type=%s) ...",
            model_size, device, compute_type,
        )
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            # Limit CPU threads to avoid starving other pipeline stages.
            cpu_threads=4,
        )
        logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio_data: sr.AudioData) -> str:
        """Transcribe raw audio data to text.

        Parameters

        audio_data : sr.AudioData
            Raw audio captured by the SpeechRecognition library.

        Returns

        str
            The transcribed text, stripped of leading/trailing whitespace.
            Returns an empty string if transcription fails or produces no
            recognizable speech.
        """
        # Convert SpeechRecognition AudioData to a WAV byte stream that faster-whisper can consume.
        wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
        wav_buffer = io.BytesIO(wav_bytes)

        try:
            segments, info = self._model.transcribe(
                wav_buffer,
                beam_size=1,          # Greedy decoding for speed.
                best_of=1,
                language="en",        # Skip language detection overhead.
                vad_filter=True,      # Filter silence segments.
                without_timestamps=True,
            )
            text = " ".join(segment.text for segment in segments).strip()
            return text
        except Exception:
            logger.exception("Transcription failed.")
            return ""



# OllamaClient -- streaming LLM interface


class OllamaClient:
    """Streams tokens from a locally-running Ollama instance.

    Uses the ``/api/generate`` endpoint with ``stream=True`` to yield tokens
    incrementally, which is essential for achieving sub-2-second latency.
    """

    def __init__(
        self,
        api_url: str = OLLAMA_API_URL,
        model: str = OLLAMA_MODEL,
    ):
        self._api_url = api_url
        self._model = model
        self._session = requests.Session()
        # Keep-alive connections reduce per-request overhead.
        self._session.headers.update({"Connection": "keep-alive"})

    def stream_response(self, prompt: str):
        """Yield tokens from the LLM as they are generated.

        Parameters

        prompt : str
            The user's transcribed speech to send to the model.

        Yields
        ------
        str
            Individual text tokens (or small token groups) from the LLM.
        """
        payload = {
            "model": self._model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": True,
            "options": {
                # Keep temperature low for focused, concise answers.
                "temperature": 0.5,
                # Limit response length to keep latency low.
                "num_predict": 200,
            },
        }

        try:
            response = self._session.post(
                self._api_url,
                json=payload,
                stream=True,
                timeout=30,
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON line from Ollama.")
                    continue

                token = data.get("response", "")
                if token:
                    yield token

                # Ollama signals end-of-generation with "done": true.
                if data.get("done", False):
                    return

        except requests.exceptions.ConnectionError:
            logger.error(
                "Cannot connect to Ollama at %s. "
                "Ensure the Ollama server is running ('ollama serve').",
                self._api_url,
            )
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out after 30 seconds.")
        except requests.exceptions.HTTPError as exc:
            logger.error("Ollama returned an HTTP error: %s", exc)
        except Exception:
            logger.exception("Unexpected error during Ollama streaming.")

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable."""
        try:
            resp = self._session.get(
                self._api_url.replace("/api/generate", ""),
                timeout=3,
            )
            return resp.status_code == 200
        except Exception:
            return False


# ===========================================================================
# TextToSpeechEngine - pyttsx3 with a dedicated worker thread
# ===========================================================================

class TextToSpeechEngine:
    """Thread-safe TTS engine backed by pyttsx3.

    pyttsx3 is not thread-safe, so all speech synthesis calls are serialized
    through an internal queue consumed by a dedicated worker thread.  This
    also ensures that sentences are spoken in order.
    """

    # Sentinel value to signal the worker thread to shut down.
    _SHUTDOWN = object()

    def __init__(self):
        self._queue: Queue = Queue()
        self._worker = threading.Thread(
            target=self._run, name="tts-worker", daemon=True,
        )
        self._worker.start()
        logger.info("TTS engine initialized.")

    def _run(self):
        """Worker loop: consume sentences from the queue and speak them."""
        # Initialize the pyttsx3 engine inside the worker thread.  Some
        # platforms require the engine to live in the same thread that
        # calls runAndWait().
        engine = pyttsx3.init()

        # Slightly increase speaking rate for a more natural conversation
        # pace and to reduce latency between sentences.
        rate = engine.getProperty("rate")
        engine.setProperty("rate", int(rate * 1.2))

        while True:
            item = self._queue.get()
            if item is self._SHUTDOWN:
                self._queue.task_done()
                break
            try:
                engine.say(item)
                engine.runAndWait()
            except Exception:
                logger.exception("TTS playback failed for: %s", item)
            finally:
                self._queue.task_done()

    def speak(self, text: str):
        """Enqueue a sentence for TTS playback.

        This method returns immediately; the actual synthesis and playback
        happen asynchronously in the TTS worker thread.

        Parameters
        ----------
        text : str
            The sentence to speak.
        """
        if text and text.strip():
            logger.debug("TTS enqueue: %s", text.strip())
            self._queue.put(text.strip())

    def speak_filler(self, phrase: str | None = None):
        """Speak a filler phrase to cover LLM latency.

        Parameters
        ----------
        phrase : str, optional
            A specific phrase to use.  If ``None``, a random filler phrase
            is selected from ``FILLER_PHRASES``.
        """
        if phrase is None:
            phrase = random.choice(FILLER_PHRASES)
        logger.info("Playing filler phrase: '%s'", phrase)
        self.speak(phrase)

    def wait_until_done(self):
        """Block until all queued sentences have been spoken."""
        self._queue.join()

    def shutdown(self):
        """Signal the TTS worker thread to exit."""
        self._queue.put(self._SHUTDOWN)
        self._worker.join(timeout=5)


# ===========================================================================
# FillerMechanism -- 800ms watchdog for LLM latency
# ===========================================================================

class FillerMechanism:
    """Plays a filler phrase if the LLM is too slow to respond.

    After the user's speech is transcribed and sent to the LLM, a timer is
    started.  If the first LLM token does not arrive within
    ``FILLER_TIMEOUT_SECONDS``, a filler phrase (or pre-recorded .wav file)
    is played to keep the user engaged and avoid awkward silence.

    The timer is cancelled as soon as the first token arrives.
    """

    def __init__(self, tts_engine: TextToSpeechEngine):
        self._tts = tts_engine
        self._timer: threading.Timer | None = None
        self._fired = threading.Event()
        self._cancelled = False

    def start(self):
        """Start the filler watchdog timer."""
        self._fired.clear()
        self._cancelled = False
        self._timer = threading.Timer(
            FILLER_TIMEOUT_SECONDS,
            self._on_timeout,
        )
        self._timer.daemon = True
        self._timer.start()
        logger.debug("Filler timer started (%.1fs).", FILLER_TIMEOUT_SECONDS)

    def cancel(self):
        """Cancel the timer -- called when the first LLM token arrives."""
        self._cancelled = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
            logger.debug("Filler timer cancelled (first token received).")

    def has_fired(self) -> bool:
        """Return True if the filler phrase was played."""
        return self._fired.is_set()

    def _on_timeout(self):
        """Callback executed when the timer expires."""
        # Guard against a race where cancel() was called just before this
        # callback ran (timer already in-flight when cancel() was called).
        if self._cancelled:
            logger.debug("Filler timer fired but was already cancelled; skipping.")
            return

        self._fired.set()

        # Attempt to play a pre-recorded .wav filler file if available.
        wav_played = self._try_play_wav_filler()
        if not wav_played:
            # Fall back to pyttsx3 synthesized filler.
            self._tts.speak_filler()

    def _try_play_wav_filler(self) -> bool:
        """Try to play a random .wav file from the filler_audio directory.

        Returns True if a file was successfully played, False otherwise.
        Uses the built-in `wave` module and writes to a temp file for
        cross-platform playback via pyttsx3 or os-level commands.

        """
        if not FILLER_AUDIO_DIR.is_dir():
            return False

        wav_files = list(FILLER_AUDIO_DIR.glob("*.wav"))
        if not wav_files:
            return False

        wav_path = random.choice(wav_files)
        logger.info("Playing filler audio file: %s", wav_path.name)

        try:
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(
                    str(wav_path),
                    winsound.SND_FILENAME | winsound.SND_ASYNC,
                )
                return True
            elif sys.platform == "linux":
                os.system(f'aplay -q "{wav_path}" &')
                return True
            elif sys.platform == "darwin":
                os.system(f'afplay "{wav_path}" &')
                return True
            else:
                return False
        except Exception:
            logger.exception("Failed to play filler .wav file.")
            return False


# ===========================================================================
# SentenceBuffer -- accumulates tokens and flushes complete sentences
# ===========================================================================

class SentenceBuffer:
    """Buffers streaming tokens and flushes complete sentences to TTS.

    Sentence boundaries are detected by the presence of sentence-ending
    punctuation marks (``.``, ``?``, ``!``).  Each time a boundary is found,
    the accumulated text up to and including the punctuation is flushed to
    the TTS engine, allowing the first sentence to be spoken while the LLM
    is still generating subsequent text.
    """

    # Characters that signal the end of a sentence.
    SENTENCE_ENDINGS = {".", "?", "!"}

    def __init__(self, tts_engine: TextToSpeechEngine):
        self._tts = tts_engine
        self._buffer: list[str] = []

    def add_token(self, token: str):
        """Add a token to the buffer and flush if a sentence boundary is found.

        Parameters
        ----------
        token : str
            A token (or small token group) from the LLM stream.
        """
        self._buffer.append(token)

        # Check whether the latest token contains a sentence-ending character.
        if any(ch in token for ch in self.SENTENCE_ENDINGS):
            self._flush()

    def flush_remaining(self):
        """Flush any remaining buffered text after the LLM finishes."""
        if self._buffer:
            self._flush()

    def _flush(self):
        """Join buffered tokens, send the sentence to TTS, and reset."""
        sentence = "".join(self._buffer).strip()
        self._buffer.clear()
        if sentence:
            logger.info("Sentence -> TTS: %s", sentence)
            self._tts.speak(sentence)


# ===========================================================================
# VoiceAssistant - top-level orchestrator
# ===========================================================================

class VoiceAssistant:
    """Orchestrates the full STT -> LLM -> TTS pipeline.

    Runs a continuous microphone listening loop in the main thread.  When
    user speech is detected, it flows through the pipeline:

    1. Audio is transcribed by faster-whisper (SpeechToText).
    2. The transcription is sent to Ollama (OllamaClient) with streaming.
    3. A filler watchdog (FillerMechanism) ensures the user is not left
       in silence if the LLM is slow.
    4. Tokens are accumulated in a SentenceBuffer and flushed to TTS
       (TextToSpeechEngine) one sentence at a time.
    """

    def __init__(self):
        # Initialize pipeline components.
        self._stt = SpeechToText()
        self._llm = OllamaClient()
        self._tts = TextToSpeechEngine()

        # Microphone and recognizer from the SpeechRecognition library.
        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = MIC_ENERGY_THRESHOLD
        self._recognizer.pause_threshold = MIC_PAUSE_THRESHOLD
        self._recognizer.dynamic_energy_threshold = True

        # Graceful shutdown flag.
        self._running = threading.Event()
        self._running.set()

    def run(self):
        """Start the voice assistant main loop.

        Blocks indefinitely until interrupted by Ctrl+C or a call to
        ``shutdown()``.
        """
        # Verify that Ollama is reachable before entering the loop.
        if not self._llm.is_available():
            logger.error(
                "Ollama server is not reachable at %s. "
                "Please start it with 'ollama serve' and pull a model "
                "('ollama pull %s') before running this script.",
                OLLAMA_API_URL, OLLAMA_MODEL,
            )
            sys.exit(1)

        logger.info("Ollama server is reachable. Model: %s", OLLAMA_MODEL)
        logger.info("Voice assistant is ready. Speak into the microphone.")
        logger.info("Press Ctrl+C to exit.\n")

        try:
            with sr.Microphone(sample_rate=16000) as source:
                # Calibrate the recognizer for ambient noise.
                logger.info("Calibrating for ambient noise (1 second) ...")
                self._recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info(
                    "Calibration complete. Energy threshold: %.0f",
                    self._recognizer.energy_threshold,
                )

                while self._running.is_set():
                    self._listen_and_respond(source)

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _listen_and_respond(self, source: sr.Microphone):
        """Listen for a single user utterance, process it, and respond.

        Parameters
        ----------
        source : sr.Microphone
            An open microphone source.
        """
        # ----- Step 1: Listen for speech ---------------------------------
        try:
            logger.info("Listening ...")
            audio = self._recognizer.listen(source, timeout=None)
        except sr.WaitTimeoutError:
            return

        # ----- Step 2: Transcribe with faster-whisper --------------------
        t0 = time.perf_counter()
        user_text = self._stt.transcribe(audio)
        stt_ms = (time.perf_counter() - t0) * 1000

        if not user_text:
            logger.info("No speech detected (empty transcription). Skipping.")
            return

        logger.info("STT (%.0f ms): %s", stt_ms, user_text)

        # ----- Step 3: Stream LLM response with filler fallback ----------
        filler = FillerMechanism(self._tts)
        sentence_buffer = SentenceBuffer(self._tts)

        filler.start()
        first_token_received = False
        full_response_tokens: list[str] = []

        t1 = time.perf_counter()

        for token in self._llm.stream_response(user_text):
            if not first_token_received:
                first_token_received = True
                filler.cancel()
                ttft_ms = (time.perf_counter() - t1) * 1000
                logger.info("Time to first token: %.0f ms", ttft_ms)

            full_response_tokens.append(token)
            sentence_buffer.add_token(token)

        # If the LLM returned nothing at all, cancel the filler and inform.
        if not first_token_received:
            filler.cancel()
            logger.warning("LLM returned no tokens.")
            self._tts.speak("I am sorry, I could not generate a response.")
            self._tts.wait_until_done()
            return

        # Flush any remaining partial sentence.
        sentence_buffer.flush_remaining()

        full_response = "".join(full_response_tokens).strip()
        total_ms = (time.perf_counter() - t0) * 1000
        logger.info("Full LLM response: %s", full_response)
        logger.info("Total pipeline time: %.0f ms", total_ms)

        # Wait for TTS to finish speaking all queued sentences before
        # returning to the listening state.
        self._tts.wait_until_done()

    def shutdown(self):
        """Cleanly shut down the assistant and all worker threads."""
        logger.info("Shutting down voice assistant ...")
        self._running.clear()
        self._tts.shutdown()
        logger.info("Voice assistant stopped.")


# ===========================================================================
# Entry Point
# ===========================================================================

def main():
    """Parse arguments and start the voice assistant."""
    # Register SIGINT handler so Ctrl+C results in a clean shutdown.
    # On Windows, signal.SIGINT is delivered to the main thread.
    assistant = VoiceAssistant()

    def _signal_handler(signum, frame):
        logger.info("Received interrupt signal. Exiting ...")
        assistant.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    assistant.run()


if __name__ == "__main__":
    main()
