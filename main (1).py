import os
import torch
import whisper
import sounddevice as sd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from pydub import AudioSegment
import speech_recognition as sr
from scipy.io.wavfile import write
import string
import base64
import easyocr
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import traceback
import time
import cv2
from moviepy import ImageSequenceClip,VideoFileClip
import imageio
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
os.environ["WANDB_MODE"] = "disabled"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


app = Flask(__name__)
CORS(app)

# Set FFmpeg path (update this if needed)
AudioSegment.converter = r"K:\ffmpeg\bin\ffmpeg.exe"

# Initialize speech recognition and Whisper model
r = sr.Recognizer()
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base")

samplerate = 16000 
duration = 5  # seconds
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

gif_directory = r"D:\sem8\NLP\package\main\ISL_Gifs"  # Folder for ISL phrase-based GIFs
letters_directory = r"D:\sem8\NLP\package\main\letters"  # Folder for letter-based JPG images

# ISL GIFs and alphabets mapping
isl_gif = [
    'any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 
    'be careful', 'can we meet tomorrow', 'did you book tickets', 'did you finish homework',
    'do you go to office', 'do you have money', 'do you want something to drink',
    'do you want tea or coffee', 'do you watch TV', 'dont worry', 'flower is beautiful',
    'good afternoon', 'good evening', 'good morning', 'good night', 'good question', 
    'had your lunch', 'happy journey', 'hello what is your name', 'how many people are there in your family',
    'i am a clerk', 'i am bore doing nothing', 'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 
    'i dont understand anything', 'i go to a theatre', 'i love to shop', 'i had to say something but i forgot', 
    'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
    'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call me later',
    'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 'please wait for sometime',
    'shall I help you', 'shall we go together tomorrow', 'sign language interpreter', 'sit down', 'stand up', 'take care',
    'there was traffic jam', 'wait I am thinking', 'what are you doing', 'what is the problem', 'what is todays date', 
    'what is your father do', 'what is your job', 'what is your mobile number', 'what is your name', 'whats up', 
    'when is your interview', 'when we will go', 'where do you stay', 'where is the bathroom', 'where is the police station', 
    'you are wrong'
]

alphabets = list("abcdefghijklmnopqrstuvwxyz")

def clean_text(text):
    text = text.lower()
    # Remove punctuation and symbols (keep only letters, numbers, and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()    
    return text

tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")

def correct_grammar(text):
    input_text = "grammar: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

def translate_to_english(text, src_lang="fr"):  # example: French to English
    # model_name = f"Helsinki-NLP/opus-mt-{src_lang}-en"
    model_name ="Helsinki-NLP/opus-mt-ROMANCE-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**tokens)
    english_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return english_text


def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    lang = detect_language(text)
    translated_text = translate_to_english(text, src_lang=lang)
    return jsonify({"translated": translated_text})

# Function to transcribe using Google API
def transcribe_google(audio_path, language="en-US"):
    with sr.AudioFile(audio_path) as source:
        audio_data = r.record(source)
        try:
            return r.recognize_google(audio_data, language=language)
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError:
            return "Error connecting to Google API."

# Function to transcribe using Whisper
def transcribe_whisper(audio_path):
    model = whisper.load_model("base")  # Or your selected Whisper model
    result = model.transcribe(audio_path, language='en', fp16=False)
    return result['text']

# Function to convert audio to WAV
def prepare_voice_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return path
    elif ext in (".mp3", ".m4a", ".ogg", ".flac"):
        audio = AudioSegment.from_file(path, format=ext[1:])
        wav_file = os.path.splitext(path)[0] + ".wav"
        audio.export(wav_file, format="wav")
        return wav_file
    else:
        raise ValueError(f"Unsupported audio format: {ext}")

# Route to handle audio file upload and transcription
@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    method = request.form.get("method", "whisper")
    
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        wav_file = prepare_voice_file(filepath)
        if method == "google":
            text = transcribe_google(wav_file)
        else:
            text = transcribe_whisper(wav_file)
 
        # Clean and correct grammar
        text = clean_text(text)
        detected_lang = detect_language(text)
        if detected_lang != "en":
            text = translate_to_english(text, src_lang=detected_lang)
        corrected_text = correct_grammar(text)
        print("corrected text:",corrected_text)
        return jsonify({"transcription": corrected_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def save_to_text_file(text, file_path="transcription.txt"):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(text + "\n")

# Live speech transcription route
@app.route("/live_transcribe")
def live_transcribe():
    method = request.args.get("method", "whisper")
    temp_wav_file = "live_audio.wav"  # Temporary file to store recorded audio

    # Set defaults if not defined elsewhere
    samplerate = 16000  # Whisper model prefers 16000 Hz
    duration = 5  # seconds

    print("Listening ({})...".format(method))

    try:
        # Record live audio using sounddevice
        audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()
        write(temp_wav_file, samplerate, audio_data)

        # Transcribe based on the selected method
        if method == "google":
            text = transcribe_google(temp_wav_file)
        else:
            text = transcribe_whisper(temp_wav_file)
        text=clean_text(text)
        detected_lang = detect_language(text)
        if detected_lang != "en":
            text = translate_to_english(text, src_lang=detected_lang)
        corrected_text=correct_grammar(text)
        # Save transcription to a text file
        if text:
            save_to_text_file(corrected_text)
            print(f"📝 Saved (Whisper): {corrected_text}")

        return jsonify({"transcription": corrected_text})

    except Exception as e:
        print("Error during live_transcribe:")
        traceback.print_exc()  # Print full error trace
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup: Remove the temporary WAV file
        if os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)

#--------------- ISL SIGN LANGUAGE IMAGE MATCHING ---------------
def load_image(path):
    """ Load an image if it exists """
    return Image.open(path) if os.path.exists(path) else None

def get_available_gifs():
    """Retrieve available ISL GIF filenames from the directory"""
    if not os.path.exists(gif_directory):
        return set()  # Return an empty set if directory doesn't exist

    return {os.path.splitext(f)[0].replace("_", " ") for f in os.listdir(gif_directory) if f.endswith(".gif")}

def encode_image_to_base64(image_path):
    """Convert an image to base64 encoding"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def combine_letter_images_to_video(word, output_path):
    images = [load_image(os.path.join(letters_directory, f"{letter}.jpg")) for letter in word if letter in alphabets]
    images = [img for img in images if img is not None]

    if not images:
        return None

    frames = [pil_to_cv2(img.resize((200, 200))) for img in images]
    height, width, _ = frames[0].shape

    # Define video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return output_path


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def load_image(path):
    try:
        from PIL import Image
        return Image.open(path)
    except Exception as e:
        print(f"[ERROR] Failed to load image: {e}")
        return None

def resize_frame(frame, size=(200,200)):
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


@app.route("/isl/<text>")
def display_isl_gif(text):

    print("hi")
    # Clean the text and split into words.
    text = "".join([i for i in text if i not in string.punctuation])
    text = clean_text(text)
    detected_lang = detect_language(text)
    if detected_lang != "en":
        text = translate_to_english(text, src_lang=detected_lang)

    words = text.lower().split()
    all_frames = []

    # Define the frame rate for the GIF (frames per second)
    fps = 1  # Adjust this value as needed

    for word in words:
        # Add a text frame showing the word
        text_frame = np.ones((200, 200, 3), dtype=np.uint8) * 255  # white background
        cv2.putText(text_frame, word.upper(), (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 2, cv2.LINE_AA)
        all_frames.extend([text_frame] * fps)  # Show text for 1 second

        gif_path = os.path.join(gif_directory, f"{word}.gif")
        if os.path.exists(gif_path):
            try:
                clip = VideoFileClip(gif_path)
                for frame in clip.iter_frames(fps=clip.fps):
                    resized = resize_frame(frame)
                    all_frames.append(resized)
                # Add pause after existing GIF
                if all_frames:
                    pause_frame = np.ones_like(all_frames[-1]) * 255
                    all_frames.extend([pause_frame] * fps)
            except Exception as e:
                print(f"[ERROR] Failed to read frames from {gif_path}: {e}", flush=True)
            continue

        print(f"[INFO] Processing word: {word}", flush=True)
        word_frames = []

        for letter in word:
            if letter in alphabets:
                img_path = os.path.join(letters_directory, f"{letter}.jpg")
                img = load_image(img_path)
                if img:
                    frame = cv2.cvtColor(np.array(img.resize((200, 200))), cv2.COLOR_RGB2BGR)
                    word_frames.append(frame)

        all_frames.extend(word_frames)

        # Add pause after constructed word
        if word_frames:
            pause_frame = np.ones_like(word_frames[0]) * 255
            all_frames.extend([pause_frame] * fps)


    if not all_frames:
        print("[WARNING] No frames found to create GIF.", flush=True)
        return jsonify({"error": "No valid frames to generate GIF"}), 400

    # Output path for the generated GIF file
    output_path = os.path.join(UPLOAD_FOLDER, "sentence_video.gif")
    print(f"[INFO] Creating GIF with {len(all_frames)} frames at {fps} fps", flush=True)

    # Create the GIF using MoviePy; ImageSequenceClip works with a list of numpy arrays.
    try:
        clip = ImageSequenceClip(all_frames, fps=fps)
        clip.write_gif(output_path, fps=fps)
    except Exception as e:
        print(f"[ERROR] Failed to create GIF: {e}", flush=True)
        return jsonify({"error": "Failed to generate GIF"}), 500

    # Read and encode the GIF as a base64 string.
    try:
        with open(output_path, "rb") as gif_file:
            encoded = base64.b64encode(gif_file.read()).decode('utf-8')
            encoded_string = f"data:image/gif;base64,{encoded}"
            print(f"[SUCCESS] Final GIF size: {len(encoded)} bytes", flush=True)
            return jsonify({"gifs": [encoded_string]})
    except Exception as e:
        print(f"[ERROR] Could not encode GIF: {e}", flush=True)
        return jsonify({"error": "Failed to encode GIF"}),500


@app.route("/get_video/<filename>")
def get_video(filename):
    try:
        # Return the video file from the uploads folder
        return send_from_directory(UPLOAD_FOLDER, filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/braille/<text>")
def text_to_braille(text):
    text = clean_text(text)
    detected_lang = detect_language(text)
    if detected_lang != "en":
        text = translate_to_english(text, src_lang=detected_lang)
    
    braille_dict = {
        'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
        'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
        'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵', ' ': ' '
    }
    
    # Convert text to braille
    braille_text = ''.join(braille_dict.get(char.lower(), '') for char in text)
    
    # Return original + braille
    return jsonify({
        "original_text": text,
        "braille": braille_text
    })


@app.route('/upload', methods=['POST'])
def upload_image():
    root = r"D:\sem8\NLP\package\main"
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image']
    image_path =os.path.join(root, image.filename)
    image.save(image_path)
    image=Image.open(image_path)
    
    reader = easyocr.Reader(['en'])  # Initialize once if used repeatedly
    results = reader.readtext(image_path, detail=0)
    extracted_text = ' '.join(results)

    extracted_text = clean_text(extracted_text)
    corrected_text = correct_grammar(extracted_text)

    # image.close()
    # os.remove(image_path)
    
    return jsonify({
        'text': corrected_text.strip(),
        # 'braille': text_to_braille(extracted_text),
        #'sign_language': display_isl_video(extracted_text)
    })

# --------------- WEB INTERFACE ---------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # return send_from_directory(UPLOAD_FOLDER, filename)
    path="uploads/filename"
    return Image.open(path) if os.path.exists(path) else None

if __name__ == "__main__":
    extracted_text="comment allez-vous aujourd'hui ?"
    extracted_text = clean_text(extracted_text)
    detected_lang = detect_language(extracted_text)
    print(detected_lang)
    if detected_lang != "en":
        extracted_text = translate_to_english(extracted_text, src_lang=detected_lang)
    corrected_text = correct_grammar(extracted_text)
    print(corrected_text)

    app.run(debug=True)