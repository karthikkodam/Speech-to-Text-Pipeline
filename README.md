# Real-Time Speech-to-Text Transcriber

A simple and efficient web app built with Streamlit and OpenAI's Whisper model that takes an audio file (.wav/.mp3) as input, applies noise reduction, and generates an accurate transcript along with timestamps.

## 1. Objective
Build a real-time or file-based transcription system.

## 2. Project Overview

This project aims to make speech transcription accessible via a lightweight web interface. Users can upload an audio file and receive transcriptions with precise time-stamped segments using Whisper — a state-of-the-art automatic speech recognition (ASR) model developed by OpenAI.

## 3. Problem Statement

Traditional ASR systems are either too heavy to deploy or lack timestamp accuracy and noise robustness. This app solves these issues by:

- Leveraging Whisper’s multilingual and timestamped transcription capabilities.
- Reducing noise using `noisereduce`.
- Deploying on the web via Streamlit with minimal user effort.

## 4. Technical Approach

- **Model Used:** `openai/whisper-small`
- **Framework:** Streamlit for frontend and backend
- **Preprocessing:**
  - Load file with `librosa`
  - Apply noise reduction (`noisereduce`)
  - Resample to 16kHz for Whisper compatibility
- **Inference:**
  - Use Whisper’s `transcribe()` with timestamps
- **Output:**
  - Full transcript
  - Timestamped text segments
  - Audio player for playback
  - 
## 5. Notebook Reference
The full Jupyter notebook is too large to upload directly to GitHub (exceeds the 100MB limit).  
To view or download the notebook, use the link below:

Notebook : ([https://your-download-link.com](https://www.kaggle.com/code/kodamkarthik281/task-kdt01)) 

## 6. Model Hosting & Space Deployment

- Hosted on **Hugging Face Spaces** using the Streamlit SDK.
- Due to CPU-only infrastructure, initial loading can take 1–2 minutes.

[Hugging Face Space (Live Demo)](https://kodamkarthik281-audio-to-text.hf.space)

## 7. Conclusion

This project demonstrates how cutting-edge speech recognition models like Whisper can be made user-friendly via Streamlit. With added features like noise reduction and timestamping, this app is suitable for real-time meeting transcription, podcast indexing, or accessibility tools.


