import whisper
import ollama
import yt_dlp
import sys
import os


class VideoSummarizer:
    def __init__(self, url: str):
        self.url = url

    def __download_sound(self, file_name="sound") -> str:
        temp_path = "./temp/"
        os.makedirs(temp_path, exist_ok=True)
        options = {
            "format": "bestaudio/best",
            "outtmpl": f"{temp_path}{file_name}.%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(self.url, download=True)
                output_file = ydl.prepare_filename(info).replace(info["ext"], "mp3")
                return output_file
        except Exception as e:
            print(f"Download Error: {e}")
            sys.exit(1)

    def __convert_to_text(self, sound_path: str, model_name="small") -> str:
        model = whisper.load_model(model_name)
        result = model.transcribe(sound_path)
        return result["text"]

    def __ai_summarizer(self, text: str, model_name="deepseek-r1:1.5b") -> str:
        input_text = f"Hello, can you please summarize this text and don't make too short for me while maintaining the most important info and making sure it's not losing important information from the text? Also, make sure to format it to look clean and readable.\n\n{text}"

        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant skilled in summarizing text while preserving important information and formatting it cleanly.",
                    },
                    {
                        "role": "user",
                        "content": input_text,
                    },
                ],
            )

            return response["message"]["content"]
        except Exception as e:
            print(f"AI summarization error: {e}")
            sys.exit(1)

    def summarize(
        self,
        sound_file_name="sound",
        stt_model="small",
        model_name="deepseek-r1:1.5b",
        output_file="output.txt",
    ) -> str:
        sound_path = self.__download_sound(sound_file_name)
        original_text = self.__convert_to_text(sound_path, stt_model)
        summarized_text = self.__ai_summarizer(original_text, model_name)

        try:
            with open(output_file, "w") as file:
                file.write(summarized_text)
                print(f"Summary saved to {output_file}")
        except Exception as e:
            print(f"Saving Error: {e}")

        return summarized_text


def get_url() -> str:
    return sys.argv[1] if len(sys.argv) > 1 else input("Enter the video url:")


if __name__ == "__main__":
    url = get_url()
    summarizer = VideoSummarizer(url)
    summarizer.summarize()
