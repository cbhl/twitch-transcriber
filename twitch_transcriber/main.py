import argparse
import signal
import sys
import threading
from queue import Queue
import streamlink
import numpy as np
from transformers import pipeline
from urllib.parse import urlparse
import subprocess
import ffmpeg
import io

def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def get_twitch_url(input_string):
    if is_url(input_string):
        return input_string
    return f"https://www.twitch.tv/{input_string}"

def signal_handler(sig, frame):
    print("\nGracefully shutting down...")
    sys.exit(0)

class AudioTranscriber:
    def __init__(self, url):
        self.url = url
        self.audio_queue = Queue()
        self.transcriber = pipeline("automatic-speech-recognition", 
                                  model="openai/whisper-base",
                                  device="cpu")
        self.should_stop = False
        
    def print_transcription(self, text):
        if text.strip():
            print(text.strip(), end=' ', flush=True)

    def process_audio_stream(self, stream_url):
        process = (
            ffmpeg
            .input(stream_url)
            .output(
                'pipe:',
                format='f32le',
                acodec='pcm_f32le',
                ac=1,
                ar='16k'
            )
            .overwrite_output()
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        CHUNK_SIZE = 1024 * 8
        audio_data = []

        while not self.should_stop:
            try:
                in_bytes = process.stdout.read(CHUNK_SIZE * 4)
                if not in_bytes:
                    break
                
                audio_chunk = np.frombuffer(in_bytes, dtype=np.float32)
                audio_data.extend(audio_chunk)

                if len(audio_data) >= 16000 * 3:
                    audio_array = np.array(audio_data, dtype=np.float32)
                    result = self.transcriber(audio_array)
                    self.print_transcription(result["text"])
                    audio_data = []

            except Exception as e:
                print(f"\nError processing audio: {e}")
                break

        process.kill()
        
    def stream_audio(self):
        try:
            streams = streamlink.streams(self.url)
            if not streams:
                print("No streams found")
                return

            stream = streams.get('audio_only', streams.get('worst'))
            if not stream:
                print("No suitable stream found")
                return

            print(f"Streaming started from {self.url}")
            
            self.process_audio_stream(stream.url)

        except KeyboardInterrupt:
            self.should_stop = True
            print("\nTranscription stopped by user")
        except Exception as e:
            print(f"\nError: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Twitch stream audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s fuslie
  %(prog)s https://www.twitch.tv/fuslie
        """)
    parser.add_argument("channel", 
                       help="Twitch channel name or full URL")
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    url = get_twitch_url(args.channel)
    transcriber = AudioTranscriber(url)
    transcriber.stream_audio()

if __name__ == "__main__":
    main()