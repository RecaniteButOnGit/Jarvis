import os
import re
import time
import base64
import tempfile
import subprocess
import threading
import difflib
import datetime
from urllib.parse import quote

import cv2
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

# ============================================================
# CONFIG
# ============================================================

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_MODEL = "google/gemma-4-e4b"

WAKE_WORDS = [
    "jarvis",
    "jervis",
    "jarves",
    "javis",
    "jarvos",
    "pajervos",
    "pajervis",
    "travis",
    "service",
]

SLEEP_PHRASES = [
    "never mind",
    "nevermind",
    "that's all",
    "that's it",
    "thats it",
    "that's all for now",
    "thats all for now",
    "that's everything",
    "thats everything",
    "we're done",
    "were done",
    "i'm done",
    "im done",
    "all done",
    "go to sleep",
    "go back to sleep",
    "go idle",
    "go silent",
    "be quiet",
    "quiet now",
    "stop talking",
    "stop responding",
    "stop answering",
    "stop listening",
    "stop listening now",
    "stop for now",
    "pause listening",
    "pause for now",
    "mute yourself",
    "mute",
    "thanks jarvis",
    "thank you jarvis",
    "thanks that's all",
    "thanks thats all",
    "thank you that's all",
    "thank you thats all",
    "bye jarvis",
    "goodbye jarvis",
    "good night jarvis",
    "goodnight jarvis",
    "abort",
    "abort mission",
    "cancel",
    "cancel that",
    "cancel this",
    "cancel it",
    "cancel task",
    "abort task",
    "dismiss",
    "dismiss that",
    "stand down",
    "power down",
    "shut down",
    "shut it down",
    "shut up",
    "leave me alone",
    "stop now",
    "sleep",
    "idle",
]

SLEEP_AFTER_SECONDS = 18
MAX_HISTORY_MESSAGES = 12

WHISPER_MODEL = "small.en"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"

SAMPLE_RATE = 16000

# These are MAX recording windows, not hard cutoffs.
RECORD_SECONDS_IDLE = 7
RECORD_SECONDS_ACTIVE = 22

AUDIO_BLOCK_SECONDS = 0.1

# Lower = more sensitive microphone.
# Higher = ignores more quiet noise.
SPEECH_RMS_THRESHOLD = 0.008

# After speech starts, Jarvis stops recording after this much silence.
SILENCE_AFTER_SPEECH_SECONDS = 1.35

# Always record at least this long once listening starts.
MIN_RECORD_SECONDS = 0.8

PIPER_EXE = r"C:\Users\me\AppData\Local\Python\pythoncore-3.14-64\Scripts\piper.exe"
PIPER_MODEL = r"C:\Users\me\Documents\GitHub\Jarvis\voices\en_US-danny-low.onnx"

WEBCAM_INDEX = 0
AUTO_SEND_WEBCAM_IMAGES = True

# After a vision question, messages like "try now" still send webcam frames.
VISION_FOLLOWUP_SECONDS = 25

# Debug image saved beside this file.
SAVE_DEBUG_WEBCAM_FRAME = True
DEBUG_WEBCAM_FRAME_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "jarvis_last_webcam.jpg",
)

# Persistent notes/todos stored beside this file
TODO_MD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "todo.md")
NOTES_MD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notes.md")

# Desktop vision
AUTO_SEND_DESKTOP_IMAGES = True
SAVE_DEBUG_DESKTOP_FRAME = True
DEBUG_DESKTOP_FRAME_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "jarvis_last_desktop.jpg",
)

# Set this to True if you want Jarvis to speak every boot step.
SPEAK_BOOT_STEPS = False

NO_THINK_SUFFIX = "\n\nAnswer directly. Do not include reasoning, analysis, hidden thoughts, or thinking tags."

# Tool data can get chunky. Keep it inside a sane prompt size.
MAX_TOOL_DATA_CHARS = 9000

# Local models sometimes blank on final tool summarization. When false,
# Jarvis uses a deterministic generic formatter for tool results instead.
USE_LM_FOR_TOOL_FINAL_PASS = False

# ============================================================
# SOUND EFFECT CONFIG
# ============================================================

SFX_ENABLED = True

SFX_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sounds",
)

# Plays when Jarvis is actually sending confirmed speech to the AI.
SFX_TRANSCRIBING_PATH = os.path.join(
    SFX_FOLDER,
    "sent_to_ai.wav",
)

# Loops every 2 seconds while LM Studio / Piper generation is busy.
SFX_WORKING_PATH = os.path.join(
    SFX_FOLDER,
    "working.wav",
)

# Plays when Jarvis exits an active conversation.
SFX_DISENGAGED_PATH = os.path.join(
    SFX_FOLDER,
    "disengaged.wav",
)

SFX_WORKING_INTERVAL_SECONDS = 2.0

# Lets the sent-to-AI chime finish before the looping working sound can cut it off.
SFX_WORKING_INITIAL_DELAY_SECONDS = 0.45

# Vision stability. Most LM Studio vision models behave better with one fresh image.
SEND_PREVIOUS_WEBCAM_FRAME_FOR_COMPARISON = False
VISION_IMAGE_WIDTH = 640
VISION_IMAGE_HEIGHT = 360
VISION_JPEG_QUALITY = 80

# Prints the raw model preview only when the cleaner would otherwise erase it.
DEBUG_PRINT_RAW_BLANK_MODEL_OUTPUT = True

# Visual comparison behavior. These prompts should receive special handling and
# should not be answered from chat history. The OpenCV step provides neutral
# pixel-difference metadata only; it does not hardcode object names or answers.
VISION_COMPARISON_MAX_TOKENS = 180
ALLOW_TWO_IMAGE_VISION_COMPARISON = False
DIFF_MIN_CHANGED_AREA_RATIO = 0.002
DIFF_GLOBAL_LIGHTING_AREA_RATIO = 0.35

# Vision conversation behavior.
# IMPORTANT: Do not keep sending webcam frames to every normal reply after
# a vision question. Only direct vision asks, or explicit follow-ups like
# "try now" shortly after a vision ask, should use the camera.
VISION_FOLLOWUP_ENABLED = True

# Lets the user verbally stop camera use for normal conversation. Direct
# vision requests like "what am I holding" will turn it back on.
VISION_CAN_BE_VERBALLY_DISABLED = True

# ============================================================
# PERSONALITY / PROMPTS
# ============================================================

SYSTEM_PROMPT = """
You are Jarvis, a fast local voice assistant with optional webcam vision and optional desktop screenshot vision.

Style:
- Calm, polished, precise, and capable.
- Formal but not stiff.
- Lightly witty when appropriate.
- Address the user as Caleb or sir occasionally, not constantly.
- Sound like a high-end personal assistant, not a chatbot.
- Be proactive when useful.
- Keep responses short unless asked for detail. Prefer one complete sentence over starting a question you may not finish.
- If the user gives a vague follow-up like "try now", "again", "how about now", or "one more time" after a vision question, treat it as a continued vision request.

Critical output rules:
- Do not think out loud.
- Do not write reasoning.
- Do not write analysis.
- Do not write <think>.
- Do not reveal hidden thoughts.
- Answer immediately with the final spoken response only.
- Do not mention being an AI model.
- Do not reveal hidden tool commands.

Vision rules:
- You may receive webcam images.
- You may receive desktop screenshots.
- Use them only when relevant.
- If the user asks what they are holding, doing, showing, pointing at, looking at, or what changed in the room/background, use vision.
- If the user asks what changed, compare the labeled Previous webcam frame against the labeled Current webcam frame (or the labeled Previous desktop screenshot against the labeled Current desktop screenshot if desktop screenshots are provided).
- If the user says "try now", "again", "one more time", "how about now", or similar after a vision request, use the newest image.
- If the image is too dark, blurry, blocked, or unclear, say so plainly.
- Never pretend to see something clearly if the frame is unclear.

Tool rules:
- If you need current info, recent info, prices, downloads, websites, news, or exact changed facts, end your response with:
[[TOOL:web_search|your search query]]

- If the user specifically asks for Wikipedia, end your response with:
[[TOOL:wikipedia|your query]]

- If the user asks for basic local facts like the current time/date/day, end your response with:
[[TOOL:local_info|time]]
[[TOOL:local_info|date]]
[[TOOL:local_info|day_of_week]]
[[TOOL:local_info|day_of_month]]

- Only use tools when needed.
- If using a tool, first say a short natural line like:
"I'll check that."
"One moment, Caleb."
"Let me verify that."
Then append the hidden tool command.
"""

TOOL_FINAL_SYSTEM_PROMPT = """
You are Jarvis, a polished local voice assistant.

You are now in FINAL ANSWER MODE.

You have already received tool results.
Do NOT request another tool.
Do NOT output [[TOOL:...]] commands.
Do NOT say you will check, search, verify, or look anything up.
Do NOT mention hidden commands.
Do NOT mention raw JSON.

Answer the user's original question using only the provided tool data.

Style:
- Concise.
- Polished.
- Natural spoken response.
- If the data is weak, incomplete, or failed, say that plainly.
- Do not invent facts not supported by the tool data.
"""

# ============================================================
# CUDA DLL FIX
# ============================================================

CUDA_PATHS = [
    r"C:\Users\me\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\nvidia\cublas\bin",
    r"C:\Users\me\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\nvidia\cuda_runtime\bin",
    r"C:\Users\me\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\nvidia\cudnn\bin",
]

for path in CUDA_PATHS:
    if os.path.isdir(path):
        os.add_dll_directory(path)
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
        print("Added CUDA DLL path:", path)

from faster_whisper import WhisperModel

# ============================================================
# GLOBALS
# ============================================================

whisper = None
camera = None

history = []
active = False
last_active_time = 0.0

previous_frame_b64 = None
previous_frame_cv = None
last_capture_previous_frame_cv = None
last_capture_current_frame_cv = None
last_vision_time = 0.0
vision_enabled = True

previous_desktop_b64 = None
previous_desktop_cv = None
last_capture_previous_desktop_cv = None
last_capture_current_desktop_cv = None
last_desktop_time = 0.0
desktop_enabled = True
last_tool_search_query = ""
app_index = None
app_index_built_at = 0.0
last_todo_snapshot = ""
last_notes_snapshot = ""

working_sfx_thread = None
working_sfx_stop_event = None
working_sfx_lock = threading.Lock()

# ============================================================
# SOUND EFFECTS
# ============================================================

def check_sfx_files():
    if not SFX_ENABLED:
        return

    checks = [
        ("Sent-to-AI", SFX_TRANSCRIBING_PATH),
        ("Working", SFX_WORKING_PATH),
        ("Disengaged", SFX_DISENGAGED_PATH),
    ]

    for name, path in checks:
        if os.path.exists(path):
            print(f"[SFX] {name} sound ready:", path)
        else:
            print(f"[SFX] Missing {name} sound:", path)


def stop_current_sfx():
    if not SFX_ENABLED:
        return

    try:
        import winsound
        winsound.PlaySound(None, 0)
    except Exception:
        pass


def play_sfx_once(path: str, async_play: bool = True):
    if not SFX_ENABLED:
        return

    if not path or not os.path.exists(path):
        print(f"[SFX] Missing sound effect: {path}")
        return

    try:
        import winsound

        flags = winsound.SND_FILENAME

        if async_play:
            flags |= winsound.SND_ASYNC

        winsound.PlaySound(path, flags)

    except Exception as e:
        print("[SFX] Failed to play sound effect:", e)


def play_sent_to_ai_sfx(async_play: bool = True):
    print("[SFX] Playing sent-to-AI sound.")
    play_sfx_once(SFX_TRANSCRIBING_PATH, async_play=async_play)


def play_disengaged_sfx(async_play: bool = True):
    print("[SFX] Playing disengaged sound.")
    play_sfx_once(SFX_DISENGAGED_PATH, async_play=async_play)


def is_working_sfx_running() -> bool:
    global working_sfx_thread

    return working_sfx_thread is not None and working_sfx_thread.is_alive()


def _working_sfx_loop(stop_event: threading.Event):
    # winsound only plays one async sound at a time.
    # Starting the working loop instantly can cut off sent_to_ai.wav,
    # so the first working tick is delayed slightly.
    next_play_time = time.monotonic() + SFX_WORKING_INITIAL_DELAY_SECONDS

    while not stop_event.is_set():
        now = time.monotonic()

        if now >= next_play_time:
            play_sfx_once(SFX_WORKING_PATH, async_play=True)
            next_play_time = now + SFX_WORKING_INTERVAL_SECONDS

        sleep_time = max(0.01, min(0.05, next_play_time - time.monotonic()))

        if stop_event.wait(timeout=sleep_time):
            break


def start_working_sfx():
    global working_sfx_thread
    global working_sfx_stop_event

    if not SFX_ENABLED:
        return

    with working_sfx_lock:
        if is_working_sfx_running():
            return

        working_sfx_stop_event = threading.Event()

        working_sfx_thread = threading.Thread(
            target=_working_sfx_loop,
            args=(working_sfx_stop_event,),
            daemon=True,
        )

        working_sfx_thread.start()


def stop_working_sfx(stop_current_sound: bool = True):
    global working_sfx_thread
    global working_sfx_stop_event

    thread_to_join = None

    with working_sfx_lock:
        if working_sfx_stop_event is not None:
            working_sfx_stop_event.set()

        thread_to_join = working_sfx_thread

        working_sfx_thread = None
        working_sfx_stop_event = None

    if thread_to_join is not None:
        thread_to_join.join(timeout=0.5)

    if stop_current_sound:
        stop_current_sfx()

# ============================================================
# SPEECH / TTS
# ============================================================

def speak(text: str, use_working_sfx: bool = True, resume_working_after: bool = False):
    text = (text or "").strip()

    if not text:
        return

    text = remove_hidden_tool_commands(text)
    text = clean_model_text(text)

    if not text:
        return

    print("Jarvis:", text)

    wav_path = tempfile.mktemp(suffix=".wav")

    piper_cmd = [
        PIPER_EXE,
        "--model",
        PIPER_MODEL,
        "--output_file",
        wav_path,
    ]

    should_restart_after_voice = False

    try:
        if use_working_sfx:
            if not is_working_sfx_running():
                start_working_sfx()

            should_restart_after_voice = resume_working_after

        process = subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        process.communicate(text)

        if use_working_sfx:
            stop_working_sfx(stop_current_sound=True)

        if os.path.exists(wav_path):
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        else:
            print("[TTS] Piper did not create a WAV file.")

        if should_restart_after_voice:
            start_working_sfx()

    except Exception as e:
        stop_working_sfx(stop_current_sound=True)
        print("TTS error:", e)

    try:
        if os.path.exists(wav_path):
            os.remove(wav_path)
    except Exception:
        pass


def boot_log(title: str, detail: str, spoken: str | None = None):
    print(f"[JARVIS] {title}. {detail}")

    if SPEAK_BOOT_STEPS and spoken:
        speak(spoken, use_working_sfx=False)

# ============================================================
# AUDIO / WHISPER
# ============================================================

def record_audio(max_seconds: int):
    print(f"Listening for up to {max_seconds} seconds...")

    block_size = int(SAMPLE_RATE * AUDIO_BLOCK_SECONDS)
    max_blocks = max(1, int(max_seconds / AUDIO_BLOCK_SECONDS))

    silence_blocks_needed = max(
        1,
        int(SILENCE_AFTER_SPEECH_SECONDS / AUDIO_BLOCK_SECONDS),
    )

    min_blocks = max(
        1,
        int(MIN_RECORD_SECONDS / AUDIO_BLOCK_SECONDS),
    )

    recorded_blocks = []
    speech_started = False
    stopped_after_silence = False
    silent_blocks = 0
    peak_rms = 0.0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=block_size,
        ) as stream:
            for block_index in range(max_blocks):
                block, overflowed = stream.read(block_size)

                if overflowed:
                    print("[AUDIO] Input overflow warning.")

                recorded_blocks.append(block.copy())

                rms = float(np.sqrt(np.mean(np.square(block))))
                peak_rms = max(peak_rms, rms)

                if rms >= SPEECH_RMS_THRESHOLD:
                    speech_started = True
                    silent_blocks = 0
                else:
                    if speech_started:
                        silent_blocks += 1

                has_recorded_minimum = block_index >= min_blocks

                if (
                    speech_started
                    and has_recorded_minimum
                    and silent_blocks >= silence_blocks_needed
                ):
                    stopped_after_silence = True
                    break

    except Exception as e:
        print("[AUDIO] Smart recording failed:", e)
        print("[AUDIO] Falling back to fixed recording.")

        audio = sd.rec(
            int(max_seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )

        sd.wait()
        recorded_blocks = [audio]

        speech_started = True
        stopped_after_silence = False

    if recorded_blocks:
        audio = np.concatenate(recorded_blocks, axis=0)
    else:
        audio = np.zeros((int(SAMPLE_RATE * 0.2), 1), dtype="float32")

    duration = len(audio) / SAMPLE_RATE
    print(
        f"[AUDIO] Recorded {duration:.2f}s. "
        f"Peak RMS: {peak_rms:.4f}. "
        f"Speech: {speech_started}. "
        f"Stopped after silence: {stopped_after_silence}."
    )

    path = tempfile.mktemp(suffix=".wav")
    sf.write(path, audio, SAMPLE_RATE)

    return path, speech_started, stopped_after_silence


def transcribe_audio(path: str) -> str:
    global whisper

    segments, info = whisper.transcribe(
        path,
        vad_filter=True,
        beam_size=1,
        condition_on_previous_text=False,
        no_speech_threshold=0.65,
    )

    text = ""

    for segment in segments:
        text += segment.text.strip() + " "

    return text.strip()

# ============================================================
# WEBCAM
# ============================================================

def text_requests_vision_off(text: str) -> bool:
    t = (text or "").lower().strip()

    off_phrases = [
        "stop looking",
        "stop seeing",
        "stop using vision",
        "stop using the camera",
        "don't use vision",
        "do not use vision",
        "don't use the camera",
        "do not use the camera",
        "turn off vision",
        "turn the camera off",
        "camera off",
        "no camera",
        "no vision",
        "just talk",
        "i just want to talk",
        "you don't have to see",
        "you do not have to see",
        "quit looking",
    ]

    return any(phrase in t for phrase in off_phrases)


def text_requests_vision_on(text: str) -> bool:
    t = (text or "").lower().strip()

    on_phrases = [
        "use vision",
        "use the camera",
        "turn on vision",
        "camera on",
        "look at this",
        "look at me",
        "can you see",
        "what do you see",
        "what am i holding",
        "what object am i holding",
    ]

    return any(phrase in t for phrase in on_phrases)


def text_requests_desktop_off(text: str) -> bool:
    t = (text or "").lower().strip()

    off_phrases = [
        "stop looking at my screen",
        "stop looking at my desktop",
        "don't look at my screen",
        "do not look at my screen",
        "don't look at my desktop",
        "do not look at my desktop",
        "turn off desktop",
        "desktop off",
        "screen off",
        "no desktop",
        "no screen",
    ]

    return any(phrase in t for phrase in off_phrases)


def text_requests_desktop_on(text: str) -> bool:
    t = (text or "").lower().strip()

    on_phrases = [
        "look at my screen",
        "look at my desktop",
        "show you my screen",
        "show you my desktop",
        "screen on",
        "desktop on",
        "use desktop",
    ]

    return any(phrase in t for phrase in on_phrases)


def user_directly_needs_vision(text: str) -> bool:
    t = (text or "").lower().strip()

    # If the user is correcting Jarvis or explicitly saying not to look,
    # do not let words like "holding" or "my hand" accidentally trigger vision.
    negative_context_phrases = [
        "i am not",
        "i'm not",
        "not telling you",
        "not asking you",
        "stop looking",
        "don't look",
        "do not look",
        "just talk",
        "i just want to talk",
        "you don't have to see",
        "you do not have to see",
    ]

    if any(phrase in t for phrase in negative_context_phrases):
        return False

    vision_phrases = [
        "what am i holding",
        "what object am i holding",
        "what do you see",
        "look at",
        "see this",
        "what is this",
        "what's this",
        "what am i doing",
        "am i holding",
        "on camera",
        "webcam",
        "camera",
        "my hand",
        "my hands",
        "behind me",
        "in front of me",
        "what color is",
        "read this",
        "can you see",
        "do you see",
        "showing",
        "pointing at",
        "looking at",
        "what changed",
        "what has changed",
        "what's changed",
        "find any differences",
        "spot any differences",
        "see any differences",
        "what is different",
        "what's different",
        "background",
        "my room",
        "room",
        "snapshot",
        "take another snapshot",
        "take a snapshot",
    ]

    return any(phrase in t for phrase in vision_phrases)


def user_directly_needs_desktop(text: str) -> bool:
    t = (text or "").lower().strip()

    negative_context_phrases = [
        "stop looking at my screen",
        "stop looking at my desktop",
        "don't look at my screen",
        "do not look at my screen",
        "don't look at my desktop",
        "do not look at my desktop",
    ]

    if any(phrase in t for phrase in negative_context_phrases):
        return False

    desktop_phrases = [
        "on my screen",
        "on my desktop",
        "my screen",
        "my desktop",
        "screen shot",
        "screenshot",
        "take a screenshot",
        "what's on my screen",
        "what is on my screen",
        "what's on my desktop",
        "what is on my desktop",
        "what changed on my screen",
        "what changed on my desktop",
        "compare my screen",
        "compare my desktop",
    ]

    return any(phrase in t for phrase in desktop_phrases)


def user_requests_desktop_comparison(text: str) -> bool:
    t = (text or "").lower().strip()

    comparison_phrases = [
        "what changed on my screen",
        "what changed on my desktop",
        "compare my screen",
        "compare my desktop",
        "what is different on my screen",
        "what is different on my desktop",
        "what's different on my screen",
        "what's different on my desktop",
        "any differences on my screen",
        "any differences on my desktop",
    ]

    return any(phrase in t for phrase in comparison_phrases)

def user_is_vision_followup(text: str) -> bool:
    t = (text or "").lower().strip()

    # Keep this deliberately narrow. Plain "now" caused Jarvis to attach
    # webcam frames to normal conversation way too often.
    followup_phrases = [
        "try now",
        "again",
        "one more time",
        "how about now",
        "what about now",
        "look now",
        "check now",
        "can you see it now",
        "can you see this now",
        "do you see it now",
        "is this better",
        "i increased the brightness",
        "i turned on the light",
        "i moved it",
        "better view",
        "here's a better view",
        "heres a better view",
        "clearer view",
        "closer view",
        "here's a closer view",
        "heres a closer view",
    ]

    if any(phrase in t for phrase in followup_phrases):
        return True

    # Very short follow-ups after a vision request can still be vision requests.
    short_followups = {
        "now",
        "this",
        "this one",
        "here",
        "right here",
    }

    return t in short_followups


def user_needs_vision(text: str) -> bool:
    return user_directly_needs_vision(text) or user_is_vision_followup(text)


def user_requests_visual_comparison(text: str) -> bool:
    t = (text or "").lower().strip()

    comparison_phrases = [
        "what changed",
        "what has changed",
        "what's changed",
        "what is different",
        "what's different",
        "what moved",
        "find any differences",
        "spot any differences",
        "see any differences",
        "any differences",
        "compare",
        "before",
        "previous",
        "last time",
        "earlier",
        "changed in the background",
        "changed in my room",
        "background changed",
        "room changed",
        "take another snapshot",
        "another snapshot",
    ]

    return any(phrase in t for phrase in comparison_phrases)


def warmup_camera(reads: int = 15):
    global camera

    if camera is None:
        return

    for _ in range(reads):
        camera.read()
        time.sleep(0.03)


def capture_webcam_b64():
    global previous_frame_b64
    global previous_frame_cv
    global last_capture_previous_frame_cv
    global last_capture_current_frame_cv
    global camera

    if camera is None:
        print("[VISION] Camera is None.")
        return previous_frame_b64, None

    frame = None
    ok = False

    for _ in range(8):
        ok, frame = camera.read()
        time.sleep(0.03)

    if not ok or frame is None:
        print("[VISION] Failed to read webcam frame.")
        return previous_frame_b64, None

    brightness = float(frame.mean())
    print(f"[VISION] Webcam frame brightness: {brightness:.2f}")

    if SAVE_DEBUG_WEBCAM_FRAME:
        try:
            cv2.imwrite(DEBUG_WEBCAM_FRAME_PATH, frame)
            print(f"[VISION] Saved debug frame: {DEBUG_WEBCAM_FRAME_PATH}")
        except Exception as e:
            print("[VISION] Failed to save debug frame:", e)

    if brightness < 5:
        print("[VISION] Warning: webcam frame appears almost completely black.")

    frame = cv2.resize(frame, (VISION_IMAGE_WIDTH, VISION_IMAGE_HEIGHT))

    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), VISION_JPEG_QUALITY],
    )

    if not success:
        print("[VISION] Failed to encode webcam frame.")
        return previous_frame_b64, None

    current_b64 = base64.b64encode(buffer).decode("utf-8")
    old_b64 = previous_frame_b64

    last_capture_previous_frame_cv = None
    if previous_frame_cv is not None:
        last_capture_previous_frame_cv = previous_frame_cv.copy()

    last_capture_current_frame_cv = frame.copy()
    previous_frame_b64 = current_b64
    previous_frame_cv = frame.copy()

    return old_b64, current_b64


def _escape_ps_single_quoted(path: str) -> str:
    return (path or "").replace("'", "''")


def _capture_desktop_to_file(path: str) -> bool:
    if not path:
        return False

    safe_path = _escape_ps_single_quoted(path)

    script = f"""
Add-Type -AssemblyName System.Windows.Forms | Out-Null
Add-Type -AssemblyName System.Drawing | Out-Null
$screen = [System.Windows.Forms.SystemInformation]::VirtualScreen
$bitmap = New-Object System.Drawing.Bitmap $screen.Width, $screen.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($screen.Left, $screen.Top, 0, 0, $bitmap.Size)
$bitmap.Save('{safe_path}', [System.Drawing.Imaging.ImageFormat]::Jpeg)
$graphics.Dispose()
$bitmap.Dispose()
"""

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=18,
        )

        if result.returncode != 0:
            print("[DESKTOP] PowerShell screenshot failed:", (result.stderr or "").strip())
            return False

        return os.path.exists(path) and os.path.getsize(path) > 0

    except Exception as e:
        print("[DESKTOP] Screenshot error:", e)
        return False


def capture_desktop_b64():
    global previous_desktop_b64
    global previous_desktop_cv
    global last_capture_previous_desktop_cv
    global last_capture_current_desktop_cv

    tmp_path = tempfile.mktemp(suffix=".jpg")

    try:
        ok = _capture_desktop_to_file(tmp_path)
        if not ok:
            return previous_desktop_b64, None

        data = None
        with open(tmp_path, "rb") as f:
            data = f.read()

        if not data:
            return previous_desktop_b64, None

        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("[DESKTOP] Failed to decode screenshot bytes.")
            return previous_desktop_b64, None

        brightness = float(img.mean())
        print(f"[DESKTOP] Screenshot brightness: {brightness:.2f}")

        if SAVE_DEBUG_DESKTOP_FRAME:
            try:
                cv2.imwrite(DEBUG_DESKTOP_FRAME_PATH, img)
                print(f"[DESKTOP] Saved debug screenshot: {DEBUG_DESKTOP_FRAME_PATH}")
            except Exception as e:
                print("[DESKTOP] Failed to save debug screenshot:", e)

        img = cv2.resize(img, (VISION_IMAGE_WIDTH, VISION_IMAGE_HEIGHT))

        success, buffer = cv2.imencode(
            ".jpg",
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), VISION_JPEG_QUALITY],
        )

        if not success:
            print("[DESKTOP] Failed to encode screenshot.")
            return previous_desktop_b64, None

        current_b64 = base64.b64encode(buffer).decode("utf-8")
        old_b64 = previous_desktop_b64

        last_capture_previous_desktop_cv = None
        if previous_desktop_cv is not None:
            last_capture_previous_desktop_cv = previous_desktop_cv.copy()

        last_capture_current_desktop_cv = img.copy()
        previous_desktop_b64 = current_b64
        previous_desktop_cv = img.copy()

        return old_b64, current_b64

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def get_desktop_difference_hint() -> str:
    global last_capture_previous_desktop_cv
    global last_capture_current_desktop_cv

    prev = last_capture_previous_desktop_cv
    curr = last_capture_current_desktop_cv

    if prev is None or curr is None:
        return "DIFF_STATUS=no_previous_frame"

    if prev.shape != curr.shape:
        curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]))

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    prev_gray = cv2.GaussianBlur(prev_gray, (7, 7), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (7, 7), 0)

    diff = cv2.absdiff(prev_gray, curr_gray)
    mean_diff = float(np.mean(diff))

    _, thresh = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    changed_pixels = int(cv2.countNonZero(thresh))
    total_pixels = int(thresh.shape[0] * thresh.shape[1])
    changed_ratio = changed_pixels / max(1, total_pixels)

    print(
        f"[DESKTOP DIFF] Mean diff: {mean_diff:.2f}. "
        f"Changed area: {changed_ratio * 100:.2f}%."
    )

    if changed_ratio < DIFF_MIN_CHANGED_AREA_RATIO and mean_diff < 3.5:
        return (
            "DIFF_STATUS=no_significant_change; "
            f"mean_pixel_delta={mean_diff:.2f}; "
            f"changed_area_percent={changed_ratio * 100:.2f}"
        )

    if changed_ratio > DIFF_GLOBAL_LIGHTING_AREA_RATIO:
        return (
            "DIFF_STATUS=whole_frame_change; "
            f"mean_pixel_delta={mean_diff:.2f}; "
            f"changed_area_percent={changed_ratio * 100:.2f}"
        )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 80:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((area, x, y, w, h))

    if not boxes:
        return (
            "DIFF_STATUS=weak_local_change; "
            f"mean_pixel_delta={mean_diff:.2f}; "
            f"changed_area_percent={changed_ratio * 100:.2f}"
        )

    boxes.sort(reverse=True, key=lambda item: item[0])
    top = boxes[:3]

    min_x = min(x for _, x, y, w, h in top)
    min_y = min(y for _, x, y, w, h in top)
    max_x = max(x + w for _, x, y, w, h in top)
    max_y = max(y + h for _, x, y, w, h in top)

    frame_width = max(1, thresh.shape[1])
    frame_height = max(1, thresh.shape[0])

    center_x = ((min_x + max_x) / 2) / frame_width
    center_y = ((min_y + max_y) / 2) / frame_height
    width_ratio = (max_x - min_x) / frame_width
    height_ratio = (max_y - min_y) / frame_height

    return (
        "DIFF_STATUS=localized_change; "
        f"mean_pixel_delta={mean_diff:.2f}; "
        f"changed_area_percent={changed_ratio * 100:.2f}; "
        f"bbox_norm_x={min_x / frame_width:.3f}; "
        f"bbox_norm_y={min_y / frame_height:.3f}; "
        f"bbox_norm_w={width_ratio:.3f}; "
        f"bbox_norm_h={height_ratio:.3f}; "
        f"center_norm_x={center_x:.3f}; "
        f"center_norm_y={center_y:.3f}"
    )


def get_visual_difference_hint() -> str:
    """
    Compares the previous and current webcam frames and returns neutral
    pixel-difference metadata.

    This intentionally does NOT name objects, guess what moved, or hardcode
    expected answers. It only tells the vision model where pixel changes were
    detected, so the model still has to inspect the image and answer naturally.
    """
    global last_capture_previous_frame_cv
    global last_capture_current_frame_cv

    prev = last_capture_previous_frame_cv
    curr = last_capture_current_frame_cv

    if prev is None or curr is None:
        return "DIFF_STATUS=no_previous_frame"

    if prev.shape != curr.shape:
        curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]))

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    prev_gray = cv2.GaussianBlur(prev_gray, (7, 7), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (7, 7), 0)

    diff = cv2.absdiff(prev_gray, curr_gray)
    mean_diff = float(np.mean(diff))

    _, thresh = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    changed_pixels = int(cv2.countNonZero(thresh))
    total_pixels = int(thresh.shape[0] * thresh.shape[1])
    changed_ratio = changed_pixels / max(1, total_pixels)

    print(
        f"[VISION DIFF] Mean diff: {mean_diff:.2f}. "
        f"Changed area: {changed_ratio * 100:.2f}%."
    )

    if changed_ratio < DIFF_MIN_CHANGED_AREA_RATIO and mean_diff < 3.5:
        return (
            "DIFF_STATUS=no_significant_change; "
            f"mean_pixel_delta={mean_diff:.2f}; "
            f"changed_area_percent={changed_ratio * 100:.2f}"
        )

    if changed_ratio > DIFF_GLOBAL_LIGHTING_AREA_RATIO:
        return (
            "DIFF_STATUS=whole_frame_change; "
            f"mean_pixel_delta={mean_diff:.2f}; "
            f"changed_area_percent={changed_ratio * 100:.2f}"
        )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 80:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((area, x, y, w, h))

    if not boxes:
        return (
            "DIFF_STATUS=weak_local_change; "
            f"mean_pixel_delta={mean_diff:.2f}; "
            f"changed_area_percent={changed_ratio * 100:.2f}"
        )

    boxes.sort(reverse=True, key=lambda item: item[0])
    top = boxes[:3]

    min_x = min(x for _, x, y, w, h in top)
    min_y = min(y for _, x, y, w, h in top)
    max_x = max(x + w for _, x, y, w, h in top)
    max_y = max(y + h for _, x, y, w, h in top)

    frame_width = max(1, thresh.shape[1])
    frame_height = max(1, thresh.shape[0])

    center_x = ((min_x + max_x) / 2) / frame_width
    center_y = ((min_y + max_y) / 2) / frame_height
    width_ratio = (max_x - min_x) / frame_width
    height_ratio = (max_y - min_y) / frame_height

    return (
        "DIFF_STATUS=localized_change; "
        f"mean_pixel_delta={mean_diff:.2f}; "
        f"changed_area_percent={changed_ratio * 100:.2f}; "
        f"bbox_norm_x={min_x / frame_width:.3f}; "
        f"bbox_norm_y={min_y / frame_height:.3f}; "
        f"bbox_norm_w={width_ratio:.3f}; "
        f"bbox_norm_h={height_ratio:.3f}; "
        f"center_norm_x={center_x:.3f}; "
        f"center_norm_y={center_y:.3f}"
    )


def _extract_float_from_hint(hint: str, key: str, default: float | None = None) -> float | None:
    match = re.search(rf"{re.escape(key)}=([0-9.]+)", hint or "")
    if not match:
        return default

    try:
        return float(match.group(1))
    except Exception:
        return default


def _horizontal_region(value: float | None) -> str:
    if value is None:
        return "an unclear part of the frame"
    if value < 0.33:
        return "the left side of the frame"
    if value > 0.67:
        return "the right side of the frame"
    return "the middle of the frame"


def _vertical_region(value: float | None) -> str:
    if value is None:
        return ""
    if value < 0.33:
        return " near the top"
    if value > 0.67:
        return " near the bottom"
    return " around mid-height"


def visual_comparison_fallback_reply(hint: str | None = None) -> str:
    hint = hint if hint is not None else get_visual_difference_hint()
    hint_lower = hint.lower()

    if "no_previous_frame" in hint_lower:
        return "I do not have a previous frame to compare against yet, sir."

    changed_percent = _extract_float_from_hint(hint, "changed_area_percent", 0.0) or 0.0
    mean_delta = _extract_float_from_hint(hint, "mean_pixel_delta", 0.0) or 0.0

    if "no_significant_change" in hint_lower:
        return "I do not see a significant pixel-level change between the two frames."

    if "whole_frame_change" in hint_lower:
        return (
            f"I detected a broad whole-frame visual change, about {changed_percent:.1f}% of the frame, "
            "but I cannot identify a specific moved object without guessing."
        )

    if "localized_change" in hint_lower or "weak_local_change" in hint_lower:
        center_x = _extract_float_from_hint(hint, "center_norm_x", None)
        center_y = _extract_float_from_hint(hint, "center_norm_y", None)
        region = _horizontal_region(center_x) + _vertical_region(center_y)
        return (
            f"I detected a localized visual change around {region}, covering about {changed_percent:.1f}% of the frame. "
            "The vision model did not identify the object cleanly, so I will not name what moved."
        )

    return (
        f"I detected pixel-level change, with an average delta of {mean_delta:.1f} and about {changed_percent:.1f}% changed area, "
        "but I cannot identify the object cleanly from that alone."
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def remove_hidden_tool_commands(reply: str) -> str:
    pattern = r"\[\[TOOL:(web_search|wikipedia|local_info)\|(.*?)\]\]"
    return re.sub(pattern, "", reply or "", flags=re.IGNORECASE | re.DOTALL).strip()


def clean_model_text(text: str) -> str:
    """
    Removes tool commands, role prefixes, and thinking tags without accidentally
    erasing the entire answer when a local model emits a malformed <think> block.
    """
    text = (text or "").replace("\x00", "")
    text = remove_hidden_tool_commands(text)

    # Remove complete reasoning blocks first.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Some local models emit "<think>" without a closing tag. The old cleaner
    # deleted everything after that, which could turn a valid response into blank.
    if re.search(r"<think>", text, flags=re.IGNORECASE):
        final_marker = re.search(
            r"(?:final answer|final|assistant|jarvis|answer)\s*:\s*(.+)$",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if final_marker:
            text = final_marker.group(1)
        else:
            text_without_tags = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
            paragraphs = [
                paragraph.strip()
                for paragraph in re.split(r"\n\s*\n", text_without_tags)
                if paragraph.strip()
            ]

            if len(paragraphs) >= 2:
                text = paragraphs[-1]
            else:
                # Do not let the TTS read raw chain-of-thought-ish text.
                if re.search(
                    r"\b(we need|the user|i should|reasoning|analysis|analyze|hidden thought)\b",
                    text_without_tags,
                    flags=re.IGNORECASE,
                ):
                    return ""

                text = text_without_tags

    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = remove_hidden_tool_commands(text)

    bad_prefixes = [
        "Assistant:",
        "Final answer:",
        "Final:",
        "Answer:",
        "Jarvis:",
    ]

    stripped = text.strip()

    for prefix in bad_prefixes:
        if stripped.lower().startswith(prefix.lower()):
            stripped = stripped[len(prefix):].strip()

    # Remove accidental markdown/code fences that sound awful in TTS.
    stripped = stripped.strip("` \n\t")

    return stripped.strip()

# ============================================================
# MEMORY
# ============================================================

def add_history(role: str, content: str):
    content = (content or "").strip()

    if not content:
        return

    content = clean_model_text(content)

    if not content:
        return

    history.append(
        {
            "role": role,
            "content": content,
        }
    )

    while len(history) > MAX_HISTORY_MESSAGES:
        history.pop(0)

# ============================================================
# WAKE / SLEEP
# ============================================================

def contains_wake_word(transcript: str) -> bool:
    t = transcript.lower()
    return any(word in t for word in WAKE_WORDS)


def remove_wake_words(transcript: str) -> str:
    cleaned = transcript.lower()

    for word in WAKE_WORDS:
        cleaned = cleaned.replace(word, "")

    return cleaned.strip(" ,.!?-")


def extract_post_wake_text(transcript: str) -> str:
    """
    Returns only the portion of the transcript after the first detected wake word.

    This prevents Jarvis from reacting to unrelated speech that happened before
    the wake word during idle listening.
    """
    raw = transcript or ""
    lower = raw.lower()

    best_index = None
    best_len = 0

    for word in WAKE_WORDS:
        idx = lower.find(word)
        if idx == -1:
            continue
        if best_index is None or idx < best_index:
            best_index = idx
            best_len = len(word)

    if best_index is None:
        return remove_wake_words(raw)

    post = raw[best_index + best_len :]
    return post.strip(" ,.!?-")


def should_sleep(transcript: str) -> bool:
    t = transcript.lower()
    return any(phrase in t for phrase in SLEEP_PHRASES)


def exit_conversation(reason: str = ""):
    global active

    if active:
        active = False

        if reason:
            print(f"[STATE] Conversation disengaged: {reason}")
        else:
            print("[STATE] Conversation disengaged.")

        play_disengaged_sfx(async_play=True)

# ============================================================
# LM STUDIO
# ============================================================

def patch_no_think(messages):
    patched_messages = []

    for message in messages:
        copied = dict(message)

        if copied.get("role") == "user":
            content = copied.get("content")

            if isinstance(content, str):
                copied["content"] = content + NO_THINK_SUFFIX

            elif isinstance(content, list):
                new_content = []

                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        new_item = dict(item)
                        new_item["text"] = new_item.get("text", "") + NO_THINK_SUFFIX
                        new_content.append(new_item)
                    else:
                        new_content.append(item)

                copied["content"] = new_content

        patched_messages.append(copied)

    return patched_messages


def call_lmstudio(messages, max_tokens=90, temperature=0.2) -> str:
    patched_messages = patch_no_think(messages)

    payload = {
        "model": LM_MODEL,
        "messages": patched_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "stop": [
            "\nUser:",
            "\nHuman:",
            "\nAssistant:",
        ],
    }

    response = requests.post(
        LM_STUDIO_URL,
        json=payload,
        timeout=120,
    )

    if response.status_code != 200:
        print("LM Studio status:", response.status_code)
        print("LM Studio response:", response.text)
        response.raise_for_status()

    data = response.json()

    try:
        raw_reply = data["choices"][0]["message"].get("content", "")
    except Exception:
        print("Unexpected LM Studio JSON:")
        print(data)
        return ""

    cleaned_reply = clean_model_text(raw_reply)

    if DEBUG_PRINT_RAW_BLANK_MODEL_OUTPUT and raw_reply.strip() and not cleaned_reply:
        print("[LM] Model returned text, but it was not safe/useful after cleaning.")
        print("[LM] Raw preview:")
        print(raw_reply[:900])

    return cleaned_reply


def looks_like_truncated_reply(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 24:
        return False

    if t.endswith(("-", "—", "…", "...", ",", ";", ":", "(", "[")):
        return True

    last_word = re.sub(r"[^a-zA-Z]+", "", t.split()[-1].lower())
    if last_word in {"and", "but", "so", "because", "to", "with", "without", "or"}:
        return True

    # If it is long-ish and doesn't end with normal sentence punctuation, it may be cut.
    if len(t) >= 50 and not re.search(r"[.!?][\"')\\]]?$", t):
        return True

    return False


def should_send_previous_frame(user_text: str) -> bool:
    if SEND_PREVIOUS_WEBCAM_FRAME_FOR_COMPARISON:
        return True

    return user_requests_visual_comparison(user_text)


def build_user_content(
    user_text: str,
    include_vision: bool,
    include_desktop: bool = False,
    force_single_current_frame: bool = False,
):
    if not include_vision and not include_desktop:
        return user_text

    if include_desktop:
        prev_img, curr_img = capture_desktop_b64()
    else:
        prev_img, curr_img = capture_webcam_b64()

    content = [
        {
            "type": "text",
            "text": user_text,
        }
    ]

    wants_comparison = (
        user_requests_desktop_comparison(user_text)
        if include_desktop
        else user_requests_visual_comparison(user_text)
    )

    comparison_mode = (
        prev_img
        and curr_img
        and not force_single_current_frame
        and (wants_comparison or should_send_previous_frame(user_text))
    )

    if comparison_mode:
        if include_desktop:
            diff_hint = get_desktop_difference_hint()
            print("[DESKTOP DIFF]", diff_hint)
        else:
            diff_hint = get_visual_difference_hint()
            print("[VISION DIFF]", diff_hint)

        content.append(
            {
                "type": "text",
                "text": (
                    "This is a visual comparison request. "
                    "Use the pixel-difference metadata only as a location hint, then inspect the current image. "
                    "If you cannot identify the exact change visually, say that clearly instead of guessing. "
                    f"Pixel-difference metadata: {diff_hint}"
                ),
            }
        )

        if ALLOW_TWO_IMAGE_VISION_COMPARISON:
            label_prev = "Previous desktop screenshot:" if include_desktop else "Previous webcam frame:"
            label_curr = "Current desktop screenshot:" if include_desktop else "Current webcam frame:"

            print("[VISION] Sending previous + current frames for comparison.")
            content.append({"type": "text", "text": label_prev})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{prev_img}"
                    },
                }
            )
            content.append({"type": "text", "text": label_curr})
        else:
            label_curr = "Current desktop screenshot:" if include_desktop else "Current webcam frame:"
            print("[VISION] Sending current frame + OpenCV diff hint for comparison.")
            content.append({"type": "text", "text": label_curr})

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{curr_img}"
                },
            }
        )

    elif curr_img:
        label_curr = "Current desktop screenshot:" if include_desktop else "Current webcam frame:"
        content.append(
            {
                "type": "text",
                "text": label_curr,
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{curr_img}"
                },
            }
        )

    return content


def should_include_vision(user_text: str) -> bool:
    global last_vision_time
    global vision_enabled

    if not AUTO_SEND_WEBCAM_IMAGES:
        return False

    if VISION_CAN_BE_VERBALLY_DISABLED and text_requests_vision_off(user_text):
        vision_enabled = False
        last_vision_time = 0.0
        print("[VISION] Vision disabled by user speech.")
        return False

    direct_vision_request = user_directly_needs_vision(user_text)

    if direct_vision_request or text_requests_vision_on(user_text):
        vision_enabled = True

    if not vision_enabled:
        return False

    recent_vision_context = (
        VISION_FOLLOWUP_ENABLED
        and last_vision_time > 0
        and time.time() - last_vision_time <= VISION_FOLLOWUP_SECONDS
    )

    explicit_followup = user_is_vision_followup(user_text)
    include_vision = direct_vision_request or (explicit_followup and recent_vision_context)

    if include_vision:
        last_vision_time = time.time()
    elif explicit_followup and not recent_vision_context:
        print("[VISION] Follow-up-like phrase heard, but vision context expired.")

    return include_vision


def should_include_desktop(user_text: str) -> bool:
    global last_desktop_time
    global desktop_enabled

    if not AUTO_SEND_DESKTOP_IMAGES:
        return False

    if text_requests_desktop_off(user_text):
        desktop_enabled = False
        last_desktop_time = 0.0
        print("[DESKTOP] Desktop vision disabled by user speech.")
        return False

    direct_desktop_request = user_directly_needs_desktop(user_text)

    if direct_desktop_request or text_requests_desktop_on(user_text):
        desktop_enabled = True

    if not desktop_enabled:
        return False

    recent_context = (
        VISION_FOLLOWUP_ENABLED
        and last_desktop_time > 0
        and time.time() - last_desktop_time <= VISION_FOLLOWUP_SECONDS
    )

    explicit_followup = user_is_vision_followup(user_text)
    include_desktop = direct_desktop_request or (explicit_followup and recent_context)

    if include_desktop:
        last_desktop_time = time.time()

    return include_desktop


def get_vision_rescue_response(user_text: str) -> str:
    print("[VISION] Retrying with a fresh single-frame vision prompt.")

    user_content = build_user_content(
        user_text,
        include_vision=True,
        force_single_current_frame=not user_requests_visual_comparison(user_text),
    )

    rescue_messages = [
        {
            "role": "system",
            "content": """
You are Jarvis, a concise webcam vision assistant.
Look at the provided webcam frame or labeled frame pair and answer the user's question.
If Previous and Current frames are provided, compare them. Do not guess a specific object unless it is visually clear.
Say only the final spoken answer.
Do not use <think>.
Do not describe your reasoning.
If the image is unclear, say it is unclear.
""",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    return call_lmstudio(
        rescue_messages,
        max_tokens=100,
        temperature=0.1,
    )


def get_jarvis_response(user_text: str) -> str:
    include_desktop = should_include_desktop(user_text)
    include_vision = should_include_vision(user_text)

    # Prefer desktop screenshots when explicitly requested, unless the user is
    # clearly asking for the webcam/camera.
    if include_desktop and not user_directly_needs_vision(user_text):
        include_vision = False

    comparison_mode = (
        (include_desktop and (user_requests_desktop_comparison(user_text) or user_requests_visual_comparison(user_text)))
        or (include_vision and user_requests_visual_comparison(user_text))
    )

    if include_desktop:
        print("[DESKTOP] Attaching desktop screenshot to this request.")
    elif include_vision:
        print("[VISION] Attaching webcam frame to this request.")

    if comparison_mode:
        print("[VISION] Visual comparison mode enabled.")

    user_content = build_user_content(user_text, include_vision, include_desktop=include_desktop)

    system_prompt = SYSTEM_PROMPT

    if comparison_mode:
        system_prompt = SYSTEM_PROMPT + """

Visual comparison override:
- The current request is about what changed.
- Compare the labeled Previous and Current images provided (webcam frames or desktop screenshots).
- Ignore old chat history for visual details.
- Name the most likely visible change in one short sentence.
- Do not claim a specific object changed unless it is visually clear.
- If you cannot identify a specific change, say you cannot tell clearly.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        *(history if not comparison_mode else []),
        {
            "role": "user",
            "content": user_content,
        },
    ]

    reply = call_lmstudio(
        messages,
        max_tokens=VISION_COMPARISON_MAX_TOKENS if comparison_mode else (200 if (include_vision or include_desktop) else 180),
        temperature=0.12 if comparison_mode else 0.2,
    )

    if reply:
        if looks_like_truncated_reply(reply):
            print("[LM] Reply looked truncated. Requesting a short continuation.")
            continuation = call_lmstudio(
                messages
                + [
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": "Continue and finish your previous reply. Output only the continuation, no preamble.",
                    },
                ],
                max_tokens=80,
                temperature=0.15,
            )

            continuation = (continuation or "").strip()
            if continuation:
                combined = (reply.rstrip() + " " + continuation.lstrip()).strip()
                combined = clean_model_text(combined)
                if combined:
                    return combined

        return reply

    print("LM Studio returned blank. Retrying simple prompt...")

    # Vision models are especially prone to blanking when context gets chunky.
    # Use a fresh single frame before giving up.
    if include_desktop:
        reply = get_desktop_rescue_response(user_text)

        if reply:
            return reply

        if comparison_mode:
            print("[DESKTOP DIFF] LM Studio blanked; using deterministic visual diff fallback.")
            return visual_comparison_fallback_reply(get_desktop_difference_hint())

    if include_vision:
        reply = get_vision_rescue_response(user_text)

        if reply:
            return reply

        if comparison_mode:
            print("[VISION DIFF] LM Studio blanked; using deterministic visual diff fallback.")
            return visual_comparison_fallback_reply()

    simple_messages = [
        {
            "role": "system",
            "content": """
You are Jarvis, a polished personal assistant.
Reply briefly and naturally.
Do not think out loud.
Do not use <think>.
Do not explain reasoning.
If the user message is incomplete, say you missed the rest and ask them to repeat it.
If images are included, answer using the image. If two labeled images are included, compare Previous versus Current.
""",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    reply = call_lmstudio(
        simple_messages,
        max_tokens=140,
        temperature=0.15,
    )

    if reply:
        if looks_like_truncated_reply(reply):
            continuation = call_lmstudio(
                simple_messages
                + [
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": "Continue and finish your previous reply. Output only the continuation, no preamble.",
                    },
                ],
                max_tokens=70,
                temperature=0.15,
            )
            continuation = (continuation or "").strip()
            if continuation:
                combined = clean_model_text((reply.rstrip() + " " + continuation.lstrip()).strip())
                if combined:
                    return combined
        return reply

    return fallback_reply(user_text)


def get_desktop_rescue_response(user_text: str) -> str:
    print("[DESKTOP] Retrying with a fresh single desktop screenshot prompt.")

    user_content = build_user_content(
        user_text,
        include_vision=False,
        include_desktop=True,
        force_single_current_frame=not (
            user_requests_desktop_comparison(user_text) or user_requests_visual_comparison(user_text)
        ),
    )

    rescue_messages = [
        {
            "role": "system",
            "content": """
You are Jarvis, a concise desktop screenshot assistant.
Look at the provided desktop screenshot or labeled screenshot pair and answer the user's question.
If Previous and Current screenshots are provided, compare them. Do not guess unless it is visually clear.
Say only the final spoken answer.
Do not use <think>.
Do not describe your reasoning.
If the image is unclear, say it is unclear.
""",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    reply = call_lmstudio(
        rescue_messages,
        max_tokens=100,
        temperature=0.15,
    )

    if reply:
        return reply

    return ""


def fallback_reply(user_text: str) -> str:
    t = user_text.lower().strip()

    if len(t.split()) <= 2 and t in ["can you", "could you", "will you"]:
        return "I missed the rest, sir. Could you repeat that?"

    if "test" in t:
        return "Test received. Systems are responsive."

    if "hello" in t or "hi" in t:
        return "Hello, Caleb. I am online."

    if "how are you" in t:
        return "Running smoothly, sir."

    return "I heard you, but the local model returned nothing."

# ============================================================
# TOOL COMMAND PARSING
# ============================================================

def extract_tool_command(reply: str):
    pattern = r"\[\[TOOL:(web_search|wikipedia|local_info)\|(.*?)\]\]"
    match = re.search(pattern, reply or "", flags=re.IGNORECASE | re.DOTALL)

    if not match:
        return clean_model_text(reply), "none", ""

    tool = match.group(1).lower().strip()
    query = match.group(2).strip()

    spoken = remove_hidden_tool_commands(reply)
    spoken = clean_model_text(spoken)

    if not spoken:
        spoken = "One moment, Caleb."

    return spoken, tool, query


def refers_to_previous_search(user_text: str) -> bool:
    """
    Detects commands like "use Wikipedia for that same search" without
    hardcoding any topic. The actual topic is stored in last_tool_search_query.
    """
    t = (user_text or "").lower()

    reference_patterns = [
        r"\bthat\s+same\s+search\b",
        r"\bthe\s+same\s+search\b",
        r"\bsame\s+search\b",
        r"\bthat\s+search\b",
        r"\bprevious\s+search\b",
        r"\blast\s+search\b",
        r"\bsame\s+thing\b",
        r"\bthat\s+same\s+thing\b",
        r"\bthat\s+query\b",
        r"\bprevious\s+query\b",
        r"\blast\s+query\b",
    ]

    return any(re.search(pattern, t, flags=re.IGNORECASE) for pattern in reference_patterns)


def looks_like_tool_instruction_only(text: str) -> bool:
    q = re.sub(r"[^a-zA-Z0-9 ]+", " ", text or "").lower()
    q = re.sub(r"\s+", " ", q).strip()

    if not q:
        return True

    instruction_words = {
        "now",
        "try",
        "use",
        "run",
        "do",
        "search",
        "engine",
        "tool",
        "google",
        "web",
        "wikipedia",
        "wiki",
        "instead",
        "same",
        "again",
        "please",
        "a",
        "an",
        "the",
        "on",
        "for",
        "about",
    }

    tokens = q.split()
    return all(token in instruction_words for token in tokens)


def extract_search_query_from_text(user_text: str) -> str:
    """
    Turns natural voice commands into a search topic without hardcoding topics.

    Examples:
    - "search up on Google what X is" -> "X"
    - "use Wikipedia for that same search" -> "" so the router reuses memory
    - "search for Unity 6 save location" -> "Unity 6 save location"
    """
    raw = (user_text or "").strip()

    if not raw:
        return ""

    if refers_to_previous_search(raw):
        return ""

    # Prefer quoted search terms when the user explicitly says them.
    quoted_match = re.search(r"['\"]([^'\"]{2,})['\"]", raw)
    if quoted_match:
        return quoted_match.group(1).strip()

    candidate = raw

    # Pull the part after common search verbs if present.
    extraction_patterns = [
        r"\b(?:search\s+up|search|look\s+up|google|web\s+search|internet\s+search)\b(?:\s+(?:for|about|on))?\s+(.+)$",
        r"\b(?:wikipedia|wiki)\b(?:\s+(?:for|about|on))?\s+(.+)$",
        r"\bwhat\s+is\s+(.+)$",
        r"\bwho\s+is\s+(.+)$",
        r"\bwhere\s+is\s+(.+)$",
        r"\bwhen\s+is\s+(.+)$",
        r"\bwhat\s+(.+?)\s+is\b",
        r"\bwho\s+(.+?)\s+is\b",
        r"\babout\s+(.+)$",
        r"\bon\s+(.+)$",
    ]

    for pattern in extraction_patterns:
        match = re.search(pattern, candidate, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            break

    cleanup_phrases = [
        r"\bwell\s+that'?s\s+fine\b",
        r"\bcan\s+you\b",
        r"\bcould\s+you\b",
        r"\bwould\s+you\b",
        r"\bplease\b",
        r"\bnow\b",
        r"\btry\s+to\b",
        r"\btry\b",
        r"\brun\b",
        r"\bdo\b",
        r"\buse\b",
        r"\byour\b",
        r"\blike\b",
        r"\bthing\b",
        r"\btool\b",
        r"\bsearch\s+engine\b",
        r"\bsearch\s+the\s+web\b",
        r"\blook\s+online\b",
        r"\bweb\s+search\b",
        r"\binternet\s+search\b",
        r"\bgoogle\s+search\b",
        r"\bgoogle\b",
        r"\bwikipedia\s+search\b",
        r"\bwikipedia\b",
        r"\bwiki\b",
        r"\binstead\s+of\b",
        r"\brather\s+than\b",
        r"\bsame\s+search\b",
    ]

    for pattern in cleanup_phrases:
        candidate = re.sub(pattern, " ", candidate, flags=re.IGNORECASE)

    candidate = re.sub(r"[^a-zA-Z0-9_+.#\- ]+", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" ,.!?-_")

    # Generic question/search cleanup. This avoids hardcoding any actual search topic.
    candidate = re.sub(r"^search\s+(?:for|about|on)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^(?:for|about|on)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^(?:what|who|where|when)\s+is\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^(?:what|who|where|when)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+is$", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^(?:a|an|the)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b(?:for|about|on)\b$", "", candidate, flags=re.IGNORECASE).strip()

    if looks_like_tool_instruction_only(candidate):
        return ""

    return candidate


def force_tool_if_obvious(user_text: str):
    global last_tool_search_query

    t = (user_text or "").lower()

    web_phrases = [
        "google",
        "search engine",
        "search the web",
        "web search",
        "look online",
        "internet search",
        "instead of wikipedia",
        "not wikipedia",
        "use your like search engine",
        "use your search engine",
    ]

    wiki_phrases = [
        "wikipedia",
        "wiki",
    ]

    web_keywords = [
        "latest",
        "recent",
        "today",
        "current",
        "news",
        "price",
        "release date",
        "download",
        "website",
        "update",
        "version",
        "2026",
    ]

    wants_web = any(phrase in t for phrase in web_phrases) or any(word in t for word in web_keywords)
    wants_wiki = any(phrase in t for phrase in wiki_phrases)

    query = extract_search_query_from_text(user_text)

    if not query and last_tool_search_query:
        print(f"[TOOL ROUTER] Reusing previous search query: {last_tool_search_query}")
        query = last_tool_search_query

    # Web/search-engine intent wins over Wikipedia if both are mentioned.
    if wants_web:
        return "web_search", query or user_text

    if wants_wiki:
        return "wikipedia", query or user_text

    return "none", ""

# ============================================================
# TOOLS
# ============================================================

def wikipedia_search(query: str) -> str:
    query = (query or "").strip()

    if not query:
        return "No Wikipedia query was provided."

    try:
        search_url = "https://en.wikipedia.org/w/api.php"

        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
        }

        search_response = requests.get(
            search_url,
            params=search_params,
            timeout=15,
            headers={
                "User-Agent": "JarvisLocalAssistant/1.0 (local personal assistant)"
            },
        )

        search_response.raise_for_status()
        search_data = search_response.json()

        results = search_data.get("query", {}).get("search", [])

        if not results:
            print("[WIKIPEDIA] No direct results. Falling back to web search.")
            fallback_results = web_search(f"site:en.wikipedia.org {query}")

            if fallback_results:
                return (
                    f"No direct Wikipedia API result was found, but web search returned likely Wikipedia results.\n\n"
                    f"{fallback_results}"
                )

            return f"No Wikipedia results found for: {query}"

        title = results[0]["title"]
        summary_title = title.replace(" ", "_")
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(summary_title, safe='')}"

        summary_response = requests.get(
            summary_url,
            timeout=15,
            headers={
                "User-Agent": "JarvisLocalAssistant/1.0 (local personal assistant)"
            },
        )

        summary_response.raise_for_status()
        summary_data = summary_response.json()

        extract = summary_data.get("extract", "")

        if not extract:
            print("[WIKIPEDIA] Found page but no summary. Falling back to web search.")
            fallback_results = web_search(f"site:en.wikipedia.org {title}")

            if fallback_results:
                return (
                    f"Wikipedia found {title}, but no summary was available from the API.\n\n"
                    f"{fallback_results}"
                )

            return f"Wikipedia found {title}, but no summary was available."

        page_url = summary_data.get("content_urls", {}).get("desktop", {}).get("page", "")

        result = f"Title: {title}\nSummary: {extract}"

        if page_url:
            result += f"\nURL: {page_url}"

        return result

    except Exception as e:
        print("[WIKIPEDIA] Direct Wikipedia API failed:", e)
        print("[WIKIPEDIA] Falling back to web search.")

        fallback_results = web_search(f"site:en.wikipedia.org {query}")

        if fallback_results:
            return (
                f"Direct Wikipedia search failed, but web search found likely Wikipedia results.\n\n"
                f"{fallback_results}"
            )

        return f"Wikipedia search failed: {e}"


def web_search(query: str) -> str:
    query = (query or "").strip()

    if not query:
        return "No web search query was provided."

    try:
        from duckduckgo_search import DDGS

        output_lines = []

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"No web results found for: {query}"

        for i, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            body = result.get("body", "No snippet")
            href = result.get("href", "No link")

            output_lines.append(
                f"{i}. {title}\n"
                f"{body}\n"
                f"{href}"
            )

        return "\n\n".join(output_lines)

    except Exception as e:
        print("[WEB] DuckDuckGo package search unavailable; using instant-answer fallback:", e)

    try:
        fallback_url = "https://api.duckduckgo.com/"

        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }

        response = requests.get(
            fallback_url,
            params=params,
            timeout=15,
            headers={
                "User-Agent": "JarvisLocalAssistant/1.0"
            },
        )

        response.raise_for_status()
        data = response.json()

        abstract = data.get("AbstractText", "")
        heading = data.get("Heading", "")

        if abstract:
            return f"{heading}\n{abstract}"

        related = data.get("RelatedTopics", [])
        snippets = []

        for item in related[:5]:
            if isinstance(item, dict) and item.get("Text"):
                snippets.append(item["Text"])

        if snippets:
            return "\n".join(snippets)

        return f"No useful instant-answer results found for: {query}"

    except Exception as e:
        return f"Web search failed: {e}"


def local_info(query: str) -> str:
    q = re.sub(r"[^a-z0-9_ ]+", " ", (query or "").lower()).strip()
    q = re.sub(r"\s+", " ", q)

    now = datetime.datetime.now().astimezone()

    if q in {"", "time"}:
        return now.strftime("%-I:%M %p") if "%" in "%-I" else now.strftime("%I:%M %p").lstrip("0")

    if q in {"date", "today"}:
        return now.strftime("%B %-d, %Y") if "%" in "%-d" else now.strftime("%B %d, %Y").replace(" 0", " ")

    if q in {"day_of_week", "weekday", "day"}:
        return now.strftime("%A")

    if q in {"day_of_month", "day_of_the_month"}:
        # Avoid leading zeros.
        return str(int(now.strftime("%d")))

    if q in {"datetime", "now"}:
        # Example: Friday, May 29, 2026 at 3:42 PM
        time_str = now.strftime("%-I:%M %p") if "%" in "%-I" else now.strftime("%I:%M %p").lstrip("0")
        return f"{now.strftime('%A')}, {now.strftime('%B')} {int(now.strftime('%d'))}, {now.strftime('%Y')} at {time_str}"

    return f"Unsupported local_info query: {query!r}"


def local_info_response_for_text(text: str) -> str | None:
    t = (text or "").lower().strip()

    time_phrases = [
        "what time is it",
        "tell me the time",
        "time is it",
        "current time",
        "the time",
    ]

    date_phrases = [
        "what's the date",
        "what is the date",
        "today's date",
        "todays date",
        "what day is it",
        "what day of the week is it",
        "day of the week",
        "weekday",
        "day of month",
        "day of the month",
    ]

    if t == "time" or any(p in t for p in time_phrases):
        now = datetime.datetime.now().astimezone()
        time_str = now.strftime("%-I:%M %p") if "%" in "%-I" else now.strftime("%I:%M %p").lstrip("0")
        return f"It is {time_str}."

    if any(p in t for p in date_phrases) or t in {"date", "day", "weekday"}:
        now = datetime.datetime.now().astimezone()
        time_str = now.strftime("%-I:%M %p") if "%" in "%-I" else now.strftime("%I:%M %p").lstrip("0")
        return f"Today is {now.strftime('%A')}, {now.strftime('%B')} {int(now.strftime('%d'))}, {now.strftime('%Y')}. It is {time_str}."

    return None

def run_tool(tool_name: str, query: str) -> str:
    global last_tool_search_query

    query = (query or "").strip()
    print(f"[TOOL] {tool_name}: {query}")

    if query:
        # Remember the last real topic so commands like "use Wikipedia for that
        # same search" can reuse the previous topic instead of searching for
        # the instruction itself.
        if not looks_like_tool_instruction_only(query):
            last_tool_search_query = query
            print(f"[TOOL ROUTER] Remembered search query: {last_tool_search_query}")

    if tool_name == "wikipedia":
        return wikipedia_search(query)

    if tool_name == "web_search":
        return web_search(query)

    if tool_name == "local_info":
        return local_info(query)

    return "No tool was used."


def trim_tool_data(tool_data: str) -> str:
    tool_data = (tool_data or "").strip()

    if len(tool_data) <= MAX_TOOL_DATA_CHARS:
        return tool_data

    return (
        tool_data[:MAX_TOOL_DATA_CHARS]
        + "\n\n[Tool data was truncated because it was too long.]"
    )


# ============================================================
# APP LAUNCHING (SHORTCUT SEARCH)
# ============================================================

def _run_powershell_json(script: str, timeout_seconds: int = 30):
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        if result.returncode != 0:
            print("[APPS] PowerShell failed:", (result.stderr or "").strip())
            return None

        out = (result.stdout or "").strip()
        if not out:
            return None

        import json
        return json.loads(out)

    except Exception as e:
        print("[APPS] PowerShell/JSON error:", e)
        return None


def build_app_index(force_refresh: bool = False):
    global app_index
    global app_index_built_at

    # Rebuild occasionally in case shortcuts change.
    if (not force_refresh) and app_index and (time.time() - app_index_built_at) < 300:
        return app_index

    script = r"""
$ErrorActionPreference = 'SilentlyContinue'
Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null
$paths = @(
  "$env:USERPROFILE\Desktop",
  "$env:PUBLIC\Desktop",
  "$env:APPDATA\Microsoft\Windows\Start Menu\Programs",
  "$env:ProgramData\Microsoft\Windows\Start Menu\Programs"
)
$paths = $paths | Where-Object { $_ -and (Test-Path $_) }
$wsh = New-Object -ComObject WScript.Shell

$lnks = @()
foreach ($p in $paths) {
  $lnks += Get-ChildItem -Path $p -Recurse -Filter *.lnk
}

$urls = @()
foreach ($p in $paths) {
  $urls += Get-ChildItem -Path $p -Recurse -Filter *.url
}

$items = @()
foreach ($f in $lnks) {
  $s = $wsh.CreateShortcut($f.FullName)
  $items += [pscustomobject]@{
    kind = 'lnk'
    name = $f.BaseName
    shortcut_path = $f.FullName
    target_path = $s.TargetPath
    arguments = $s.Arguments
    working_directory = $s.WorkingDirectory
  }
}

foreach ($f in $urls) {
  $raw = Get-Content -LiteralPath $f.FullName -Raw
  $m = [regex]::Match($raw, '(?im)^URL=(.+)$')
  $url = if ($m.Success) { $m.Groups[1].Value.Trim() } else { '' }
  $items += [pscustomobject]@{
    kind = 'url'
    name = $f.BaseName
    shortcut_path = $f.FullName
    target_path = $url
    arguments = ''
    working_directory = ''
  }
}

$items | ConvertTo-Json -Compress
"""

    data = _run_powershell_json(script, timeout_seconds=35)

    # ConvertTo-Json returns an object for a single item, array for many.
    if isinstance(data, dict):
        data = [data]

    app_index = data if isinstance(data, list) else []
    app_index_built_at = time.time()

    print(f"[APPS] Indexed {len(app_index)} shortcuts.")
    return app_index


# ============================================================
# TODO / NOTES (MARKDOWN) WITH DIFF
# ============================================================

def _ensure_text_file(path: str, default_text: str):
    try:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(default_text)
    except Exception as e:
        print("[FILES] Failed to ensure file:", path, e)


def _read_text(path: str) -> str:
    try:
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        print("[FILES] Read error:", path, e)
        return ""


def _write_text(path: str, text: str) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text or "")
        return True
    except Exception as e:
        print("[FILES] Write error:", path, e)
        return False


def _append_line(path: str, line: str) -> bool:
    try:
        with open(path, "a", encoding="utf-8") as f:
            if line and not line.endswith("\n"):
                line += "\n"
            f.write(line or "")
        return True
    except Exception as e:
        print("[FILES] Append error:", path, e)
        return False


def _short_diff(old: str, new: str, max_lines: int = 18) -> str:
    old_lines = (old or "").splitlines()
    new_lines = (new or "").splitlines()

    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="before",
            tofile="after",
            lineterm="",
            n=1,
        )
    )

    # Drop headers to keep TTS clean.
    diff_lines = [line for line in diff_lines if not line.startswith(("---", "+++", "@@"))]

    if not diff_lines:
        return "No changes."

    clipped = diff_lines[:max_lines]
    if len(diff_lines) > max_lines:
        clipped.append("... (diff truncated)")

    # Make it a single speakable sentence-ish chunk.
    cleaned = []
    for line in clipped:
        if line.startswith("+"):
            cleaned.append("Added: " + line[1:].strip())
        elif line.startswith("-"):
            cleaned.append("Removed: " + line[1:].strip())
        else:
            cleaned.append(line.strip())

    return " ".join([c for c in cleaned if c])


def _extract_after_prefix(text: str, prefix: str) -> str:
    m = re.match(rf"^{re.escape(prefix)}\s+(.+)$", (text or "").strip(), flags=re.IGNORECASE)
    return (m.group(1).strip() if m else "")


def handle_todo_notes_command(transcript: str) -> tuple[bool, str]:
    """
    Returns (handled, reply).

    Supported phrases (examples):
    - "add todo buy milk"
    - "add note prefers short answers"
    - "show todo"
    - "show notes"
    - "diff todo"
    - "diff notes"
    """
    global last_todo_snapshot
    global last_notes_snapshot

    t = (transcript or "").strip()
    tl = t.lower().strip()

    if not t:
        return False, ""

    _ensure_text_file(TODO_MD_PATH, "# Todo\n\n")
    _ensure_text_file(NOTES_MD_PATH, "# Notes\n\n")

    if tl.startswith("add todo "):
        item = _extract_after_prefix(t, "add todo")
        if not item:
            return True, "What should I add to your todo list, sir?"
        ok = _append_line(TODO_MD_PATH, f"- [ ] {item}")
        return True, ("Added to todo." if ok else "I could not write to todo.md.")

    if tl.startswith("add note "):
        item = _extract_after_prefix(t, "add note")
        if not item:
            return True, "What should I note down, sir?"
        ok = _append_line(NOTES_MD_PATH, f"- {item}")
        return True, ("Added to notes." if ok else "I could not write to notes.md.")

    if tl in ["show todo", "read todo", "open todo"]:
        content = _read_text(TODO_MD_PATH).strip()
        last_todo_snapshot = content
        if not content:
            return True, "Your todo list is empty."
        # Keep it brief for TTS.
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
        preview = "; ".join(lines[:8])
        if len(lines) > 8:
            preview += "; and more."
        return True, (preview if preview else "Your todo list is empty.")

    if tl in ["show notes", "read notes", "open notes"]:
        content = _read_text(NOTES_MD_PATH).strip()
        last_notes_snapshot = content
        if not content:
            return True, "Notes are empty."
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
        preview = "; ".join(lines[:8])
        if len(lines) > 8:
            preview += "; and more."
        return True, (preview if preview else "Notes are empty.")

    if tl in ["diff todo", "todo diff", "what changed in todo", "what changed in my todo"]:
        current = _read_text(TODO_MD_PATH).strip()
        diff_text = _short_diff(last_todo_snapshot, current)
        last_todo_snapshot = current
        return True, diff_text

    if tl in ["diff notes", "notes diff", "what changed in notes", "what changed in my notes"]:
        current = _read_text(NOTES_MD_PATH).strip()
        diff_text = _short_diff(last_notes_snapshot, current)
        last_notes_snapshot = current
        return True, diff_text

    return False, ""


def _score_app_match(name: str, query: str) -> int:
    n = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    q = re.sub(r"[^a-z0-9]+", " ", (query or "").lower()).strip()

    if not n or not q:
        return 0

    if n == q:
        return 100

    score = 0
    q_words = [w for w in q.split() if w]

    # All words present?
    if q_words and all(w in n for w in q_words):
        score += 40

    if n.startswith(q):
        score += 25

    if q in n:
        score += 20

    # Shorter names are often better matches when equal.
    score += max(0, 10 - min(10, len(n) // 4))

    return score


def find_best_app_shortcut(query: str):
    items = build_app_index(force_refresh=False) or []

    best = None
    best_score = 0

    for item in items:
        name = item.get("name", "")
        score = _score_app_match(name, query)
        if score > best_score:
            best = item
            best_score = score

    if best_score < 25:
        return None

    return best


def extract_open_app_query(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    m = re.match(r"^(?:open|launch|start)\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" .,!?:;")

    m = re.search(r"\bsearch\s+(?:for\s+)?(.+?)\s+and\s+(?:open|launch|start)\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" .,!?:;")

    return ""


def open_app_from_query(query: str) -> tuple[bool, str]:
    q = (query or "").strip()
    if not q:
        return False, "Which app should I open, sir?"

    item = find_best_app_shortcut(q)
    if not item:
        return False, f"I could not find a desktop or Start Menu shortcut matching {q!r}."

    shortcut_path = item.get("shortcut_path", "")
    kind = (item.get("kind") or "").lower()

    try:
        if shortcut_path and os.path.exists(shortcut_path):
            os.startfile(shortcut_path)
            return True, f"Opening {item.get('name', q)}."

        # Fallback: try to start the target directly.
        target = (item.get("target_path") or "").strip()
        args = (item.get("arguments") or "").strip()

        if kind == "url" and target:
            os.startfile(target)
            return True, f"Opening {item.get('name', q)}."

        if target:
            cmd = [target]
            if args:
                cmd += re.findall(r'\"[^\"]+\"|\\S+', args)
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, f"Opening {item.get('name', q)}."

        return False, f"I found a shortcut named {item.get('name', q)!r}, but it did not include a runnable target."

    except Exception as e:
        return False, f"I tried to open {item.get('name', q)}, but it failed: {e}"


def _clean_tool_sentence(text: str, max_chars: int = 440) -> str:
    text = re.sub(r"\s+", " ", text or "").strip(" -:;\t\n")
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip(" ,.;:-") + "..."
    return text


def _remove_duplicate_heading(heading: str, body: str) -> str:
    heading_clean = _clean_tool_sentence(heading, 120)
    body_clean = _clean_tool_sentence(body, 700)

    if not heading_clean:
        return body_clean

    pattern = re.compile(
        r"^" + re.escape(heading_clean) + r"\b\s*(?:[-–—:])?\s*",
        flags=re.IGNORECASE,
    )

    trimmed = pattern.sub("", body_clean).strip()

    # If the summary already starts naturally with the heading, say the summary
    # directly instead of making TTS say "Topic: Topic is...".
    natural_start = re.match(
        r"^" + re.escape(heading_clean) + r"\s+(?:is|are|was|were|refers|means|describes|includes)\b",
        body_clean,
        flags=re.IGNORECASE,
    )

    if natural_start:
        return body_clean

    if trimmed and trimmed != body_clean:
        return f"{heading_clean}: {trimmed}"

    return f"{heading_clean}: {body_clean}" if body_clean else heading_clean


def _score_tool_line_for_query(line: str, query: str) -> int:
    line_l = line.lower()
    words = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(w) > 1]
    score = 0

    for word in words:
        if word in line_l:
            score += 3

    if query and line_l.startswith(query.lower()):
        score += 6

    if "..." not in line:
        score += 1

    # Prefer definition-ish lines.
    if re.search(r"\b(is|are|was|were|refers to|programming language|video game|language)\b", line_l):
        score += 2

    return score


def _format_related_topic_line(query: str, line: str) -> str:
    line = _clean_tool_sentence(line, 430)
    query = _clean_tool_sentence(query, 120)

    if not line:
        return ""

    # Generic split for DuckDuckGo related-topic lines like:
    # "Rec Room (video game) A virtual reality, online video game..."
    if query:
        match = re.match(
            r"^(?P<title>" + re.escape(query) + r"(?:\s*\([^)]*\))?)\s+(?P<body>.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            title = _clean_tool_sentence(match.group("title"), 160)
            body = _clean_tool_sentence(match.group("body"), 360)
            return _remove_duplicate_heading(title, body)

    return line


def fallback_tool_answer(tool_name: str, query: str, tool_data: str) -> str:
    data = (tool_data or "").strip()

    if not data:
        return "I searched, but I did not receive any usable results."

    lines = [line.strip() for line in data.splitlines() if line.strip()]

    # Wikipedia/direct summary format.
    title_match = re.search(r"^Title:\s*(.+)$", data, flags=re.MULTILINE)
    summary_match = re.search(
        r"^Summary:\s*(.+?)(?:\nURL:|\n\d+\.|\Z)",
        data,
        flags=re.DOTALL | re.MULTILINE,
    )

    if title_match and summary_match:
        title = title_match.group(1).strip()
        summary = _clean_tool_sentence(summary_match.group(1), 430)
        return _remove_duplicate_heading(title, summary)

    # DuckDuckGo instant-answer format: heading on line 1, summary below.
    if (
        len(lines) >= 2
        and not lines[0].startswith("1.")
        and not lines[0].lower().startswith("title:")
        and not lines[0].lower().startswith("summary:")
        and not re.search(r"https?://", lines[0], flags=re.IGNORECASE)
    ):
        # If these are RelatedTopics lines, choose the line that best matches
        # the query instead of gluing unrelated meanings together.
        if len(lines) > 2 and not any(line.startswith(("http://", "https://")) for line in lines[:3]):
            best = max(lines, key=lambda item: _score_tool_line_for_query(item, query))
            if _score_tool_line_for_query(best, query) > 0:
                return _format_related_topic_line(query, best)

        heading = lines[0]
        body = _clean_tool_sentence(" ".join(lines[1:]), 430)
        if body:
            return _remove_duplicate_heading(heading, body)

    # DuckDuckGo numbered result format.
    first_result = re.search(
        r"^1\.\s*(.+?)\n(.+?)(?:\nhttps?://|\n\n2\.|\Z)",
        data,
        flags=re.DOTALL | re.MULTILINE,
    )

    if first_result:
        title = _clean_tool_sentence(first_result.group(1), 180)
        snippet = _clean_tool_sentence(first_result.group(2), 360)
        return _remove_duplicate_heading(title, snippet)

    # Plain related-topic fallback. Pick the best line, not every line.
    if lines:
        best = max(lines, key=lambda item: _score_tool_line_for_query(item, query))
        return _format_related_topic_line(query, best)

    compact = _clean_tool_sentence(data, 430)
    return compact

def get_final_tool_response(original_question: str, tool_name: str, query: str, tool_data: str) -> str:
    tool_data = trim_tool_data(tool_data)

    if not tool_data:
        return "I searched, but I did not receive any usable results."

    if not USE_LM_FOR_TOOL_FINAL_PASS:
        print("[FINAL TOOL PASS] Using deterministic tool summary formatter.")
        return fallback_tool_answer(tool_name, query, tool_data)

    print("[FINAL TOOL PASS] Feeding tool result into LM Studio.")
    print(f"[FINAL TOOL PASS] Tool data chars: {len(tool_data)}")

    messages = [
        {
            "role": "system",
            "content": TOOL_FINAL_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"""
Original user question:
{original_question}

Tool used:
{tool_name}

Tool query:
{query}

Tool results:
{tool_data}

Give the final spoken answer now.

Hard rules:
- Do not call another tool.
- Do not output [[TOOL:...]].
- Do not say "I'll check" or "let me verify."
- Use the tool results above.
""",
        },
    ]

    reply = call_lmstudio(
        messages,
        max_tokens=180,
        temperature=0.15,
    )

    reply = clean_model_text(reply)

    if not reply:
        print("[FINAL TOOL PASS] LM Studio blanked; using deterministic tool summary fallback.")
        return fallback_tool_answer(tool_name, query, tool_data)

    if "[[TOOL:" in reply:
        print("[FINAL TOOL PASS] Model tried to emit a tool command. Removing it.")
        reply = remove_hidden_tool_commands(reply)
        reply = clean_model_text(reply)

    if not reply:
        return "I found results, but the model tried to search again instead of answering."

    return reply.strip()

# ============================================================
# BOOT
# ============================================================

def boot():
    global whisper
    global camera

    check_sfx_files()

    boot_log(
        "Initializing Jarvis",
        "Primary runtime loading",
        "Initializing systems.",
    )

    boot_log(
        "Loading audio system",
        "Whisper CUDA preparing",
        "Audio loading.",
    )

    print("Loading Whisper...")
    whisper = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
    )

    boot_log(
        "Speech recognition online",
        "Whisper loaded successfully",
        "Audio online.",
    )

    boot_log(
        "Opening webcam",
        "Vision stream starting",
        "Vision loading.",
    )

    try:
        camera = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        camera.set(cv2.CAP_PROP_FPS, 30)

        warmup_camera(15)

    except Exception as e:
        print("[VISION] Camera open error:", e)
        camera = None

    if camera is not None and camera.isOpened():
        boot_log(
            "Visual input online",
            "Webcam feed acquired",
            "Vision online.",
        )
    else:
        boot_log(
            "Visual input warning",
            "Webcam unavailable",
            "Vision offline.",
        )

    boot_log(
        "Runtime ready",
        "Main loop prepared",
        "Systems ready.",
    )

    speak("Jarvis online.", use_working_sfx=False)

    print()
    print("Jarvis is running.")
    print("Say 'Jarvis' to wake it up.")
    print()

# ============================================================
# MAIN HANDLER
# ============================================================

def handle_user_message(transcript: str):
    local_reply = local_info_response_for_text(transcript)
    if local_reply:
        speak(local_reply, use_working_sfx=False)
        add_history("user", transcript)
        add_history("assistant", local_reply)
        return

    handled, reply = handle_todo_notes_command(transcript)
    if handled:
        speak(reply, use_working_sfx=False)
        add_history("user", transcript)
        add_history("assistant", reply)
        return

    open_query = extract_open_app_query(transcript)
    if open_query:
        ok, reply = open_app_from_query(open_query)
        speak(reply, use_working_sfx=False)
        add_history("user", transcript)
        add_history("assistant", reply)
        return

    forced_tool, forced_query = force_tool_if_obvious(transcript)

    if forced_tool != "none":
        speak("One moment, Caleb.", use_working_sfx=True)

        add_history("user", transcript)
        add_history("assistant", "One moment, Caleb.")

        start_working_sfx()

        try:
            tool_data = run_tool(forced_tool, forced_query)

            print("[TOOL RESULT]")
            print(tool_data)

            final_reply = get_final_tool_response(
                original_question=transcript,
                tool_name=forced_tool,
                query=forced_query,
                tool_data=tool_data,
            )

            speak(final_reply, use_working_sfx=True)
            add_history("assistant", final_reply)

        finally:
            stop_working_sfx(stop_current_sound=True)

        return

    start_working_sfx()

    try:
        raw_reply = get_jarvis_response(transcript)
        spoken_reply, tool, query = extract_tool_command(raw_reply)

        should_resume_after_speaking = tool != "none"

        speak(
            spoken_reply,
            use_working_sfx=True,
            resume_working_after=should_resume_after_speaking,
        )

        add_history("user", transcript)
        add_history("assistant", spoken_reply)

        if tool == "none":
            return

        if not query:
            query = transcript

        tool_data = run_tool(tool, query)

        print("[TOOL RESULT]")
        print(tool_data)

        final_reply = get_final_tool_response(
            original_question=transcript,
            tool_name=tool,
            query=query,
            tool_data=tool_data,
        )

        speak(final_reply, use_working_sfx=True)
        add_history("assistant", final_reply)

    finally:
        stop_working_sfx(stop_current_sound=True)

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    global active
    global last_active_time

    boot()

    while True:
        was_active_at_record_start = active
        record_seconds = RECORD_SECONDS_ACTIVE if active else RECORD_SECONDS_IDLE

        audio_path, speech_started, stopped_after_silence = record_audio(record_seconds)

        # New behavior:
        # - If already active, this speech is definitely meant for Jarvis,
        #   so play the sent-to-AI sound as soon as the recording window ends.
        # - If idle, do NOT play it yet. First transcribe and confirm wake word.
        if was_active_at_record_start and speech_started:
            if stopped_after_silence:
                print("[AUDIO] Active speech ended by silence. Starting transcription.")
            else:
                print("[AUDIO] Active speech captured. Starting transcription.")

            play_sent_to_ai_sfx(async_play=True)

        elif speech_started:
            print("[AUDIO] Idle speech captured. Transcribing silently until wake word is confirmed.")

        try:
            transcript = transcribe_audio(audio_path)
        except Exception as e:
            print("Whisper error:", e)
            continue

        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

        if not transcript:
            if active and time.time() - last_active_time > SLEEP_AFTER_SECONDS:
                print("Jarvis went idle from silence.")
                exit_conversation("timeout from silence")

            time.sleep(0.15)
            continue

        print("You:", transcript)

        if not active:
            if contains_wake_word(transcript):
                print("[WAKE] Wake word confirmed after transcription.")
                play_sent_to_ai_sfx(async_play=False)

                active = True
                last_active_time = time.time()

                cleaned = extract_post_wake_text(transcript)

                if not cleaned:
                    speak("At your service, Caleb.", use_working_sfx=False)
                    continue

                transcript = cleaned
            else:
                print("[WAKE] No wake word detected. Ignoring idle speech.")
                continue

        if should_sleep(transcript):
            speak("Standing by.", use_working_sfx=False)
            exit_conversation("sleep phrase")
            continue

        last_active_time = time.time()

        try:
            handle_user_message(transcript)
        except Exception as e:
            stop_working_sfx(stop_current_sound=True)
            print("Jarvis loop error:", e)
            speak("A subsystem fault occurred, but I remain online.", use_working_sfx=False)

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("Shutting down Jarvis.")

    finally:
        stop_working_sfx(stop_current_sound=True)

        if camera is not None:
            camera.release()
