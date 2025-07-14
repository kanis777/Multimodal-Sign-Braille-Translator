# Multimodal Accessibility Translator

A Flask-based web application that enhances accessibility by converting speech, text, and image inputs into Indian Sign Language (ISL) and Braille. The system supports live audio transcription, OCR from images, multilingual translation, grammar correction, and dynamic sign language GIF/video generation.

---

## 🔧 Features

- 🎙️ **Speech-to-Text**
  - Whisper (OpenAI) and Google Speech API integration.
  - Live recording and uploaded audio file support.
  
- 🖼️ **Image-to-Text**
  - Extracts text using EasyOCR.
  
- 📝 **Text Processing**
  - Detects language using `langdetect`.
  - Translates non-English text to English using MarianMT.
  - Corrects grammar using a pretrained T5 model.
  
- 🤟 **Sign Language Conversion**
  - Matches phrases to existing ISL GIFs.
  - Generates letter-by-letter ISL videos when no phrase match is found.
  
- 🔡 **Braille Conversion**
  - Converts cleaned, corrected text to Unicode Braille characters.

---

## 🚀 How It Works

1. **Input**: User can input data via:
   - Audio recording or upload
   - Image (containing text)
   - Direct text input

2. **Processing**:
   - Language detection and translation (if not English)
   - Grammar correction using a T5 model
   - For audio: Whisper or Google transcribes the audio to text
   - For image: EasyOCR extracts text from the image

3. **Output**:
   - Translates text to:
     - ISL GIF or dynamically generated letter videos
     - Unicode Braille characters

---

## 📂 Directory Structure
```bash
main/
├── main.py
├── 3d chars/ #blender images
├── templates/
│ └── index.html
├── uploads/
├── ISL_Gifs/ # Phrase-based GIFs
└── letters/ # A–Z letter images
```
---

---

## 🧪 APIs

| Route | Method | Description |
|-------|--------|-------------|
| `/transcribe` | POST | Transcribes uploaded audio (`google` or `whisper`) |
| `/live_transcribe` | GET | Records and transcribes 5s audio |
| `/upload` | POST | OCR from uploaded image |
| `/isl/<text>` | GET | Converts text to ISL GIF |
| `/braille/<text>` | GET | Converts text to Braille |
| `/translate` | POST | Language detection & translation to English |

---

## 🛠️ Tech Stack

- Python, Flask
- OpenAI Whisper
- Google Speech Recognition
- EasyOCR
- HuggingFace Transformers (T5 for grammar, MarianMT for translation)
- MoviePy & OpenCV
- Langdetect

---

## 📦 Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/multimodal-translator.git
   cd main/
Install requirements:

```
pip install -r requirements.txt
Set ffmpeg path:
```
Update the line:

```
AudioSegment.converter = r"K:\ffmpeg\bin\ffmpeg.exe"
```
Run the app:

```
python app.py
Access in browser: http://localhost:5000/

```

