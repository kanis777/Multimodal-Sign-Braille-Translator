# Multimodal Accessibility Translator

A Flask-based web application that enhances accessibility by converting speech, text, and image inputs into Indian Sign Language (ISL) and Braille. The system supports live audio transcription, OCR from images, multilingual translation, grammar correction, and dynamic sign language GIF/video generation.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python">
  <img src="https://img.shields.io/badge/Flask-2.3.3-green?logo=flask">
  <img src="https://img.shields.io/badge/OpenAI%20Whisper-base-orange?logo=openai">
  <img src="https://img.shields.io/badge/Google%20Speech%20API-v1-lightgrey?logo=google">
  <img src="https://img.shields.io/badge/EasyOCR-1.6.2-yellow?logo=pypi">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-ffcd00?logo=huggingface">
  <img src="https://img.shields.io/badge/T5-grammar--correction-9cf?logo=huggingface">
  <img src="https://img.shields.io/badge/MarianMT-Translation-orange?logo=huggingface">
  <img src="https://img.shields.io/badge/OpenCV-4.5.5-red?logo=opencv">
  <img src="https://img.shields.io/badge/MoviePy-1.0.3-brightgreen?logo=python">
  <img src="https://img.shields.io/badge/Pillow-PIL-yellowgreen?logo=python">
  <img src="https://img.shields.io/badge/FFmpeg-audio%20conversion-blueviolet?logo=ffmpeg">
  <img src="https://img.shields.io/badge/PyDub-Audio-yellow?logo=python">
</p>

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
2. Install requirements:

```
pip install -r requirements.txt
Set ffmpeg path:
```
3. Update the line:

```
AudioSegment.converter = r"K:\ffmpeg\bin\ffmpeg.exe"
```
4. Run the app:

```
python app.py
Access in browser: http://localhost:5000/

```

