# Video Summarizer

A Python-based tool that extracts audio from a YouTube video, transcribes it using Whisper, and summarizes the transcription using DeepSeek running locally via Ollama.

## Features

- Downloads audio from YouTube videos using `yt-dlp`
- Converts audio to text using OpenAI's `whisper`
- Summarizes the transcribed text using `ollama` with DeepSeek
- Saves the summary to a text file

## Installation

### 1. Install Dependencies

Ensure you have Python installed (3.11) and install the required libraries:

```bash
pip install -r requirements.txt
```

### 2. Install Ollama & DeepSeek Model

Follow Ollama's installation guide: [Ollama Official Site](https://ollama.com/)

Then, download the DeepSeek model:

```bash
ollama pull deepseek-r1:1.5b
```

## Usage

Run the script with a YouTube video URL:

```bash
python main.py "YOUTUBE_URL"
```

The script will:

1. Download the audio from the YouTube video
2. Transcribe the audio using Whisper
3. Summarize the transcription using DeepSeek via Ollama
4. Save the summary in `output.txt`

## Example Output

```bash
Summary saved to output.txt
```

## Customization

- Modify the Whisper model by changing `stt_model` (default: `small`)
- Use a different Ollama model by setting `model_name` in `summarize()`
- Change the output file by specifying `output_file`

##


