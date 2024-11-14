import argparse
import signal
import sys
import threading
from queue import Queue
import streamlink
import numpy as np
import sounddevice as sd
from transformers import pipeline

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
        
    def stream_audio(self):
        streams = streamlink.streams(self.url)
        if not streams:
            print("No streams found")
            return

        # Get audio-only stream if available, otherwise lowest quality
        stream_url = streams.get('audio_only', streams.get('worst')).url
        
        # Set up audio stream parameters
        CHUNK_SIZE = 1024 * 8
        SAMPLE_RATE = 16000
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            # Ensure single channel by taking mean across channels if stereo
            audio_chunk = np.mean(indata, axis=1) if indata.shape[1] > 1 else indata[:, 0]
            self.audio_queue.put(audio_chunk)

        try:
            # Explicitly specify mono input
            with sd.InputStream(callback=audio_callback,
                              channels=1,
                              samplerate=SAMPLE_RATE,
                              blocksize=CHUNK_SIZE,
                              dtype=np.float32):
                print("Streaming started. Press Ctrl+C to stop.")
                
                while True:
                    # Collect audio chunks for processing
                    audio_data = []
                    while len(audio_data) < SAMPLE_RATE * 3:  # 3 seconds of audio
                        if not self.audio_queue.empty():
                            chunk = self.audio_queue.get()
                            audio_data.extend(chunk)
                    
                    # Convert to numpy array and ensure correct shape
                    audio_array = np.array(audio_data, dtype=np.float32)
                    
                    # Ensure audio is properly shaped for the model
                    if len(audio_array.shape) == 1:
                        audio_array = audio_array.reshape(-1)
                    
                    result = self.transcriber(audio_array)
                    
                    if result["text"].strip():
                        print(result["text"])
                    
        except KeyboardInterrupt:
            print("\nTranscription stopped by user")
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe Twitch stream audio")
    parser.add_argument("url", help="Twitch stream URL")
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    transcriber = AudioTranscriber(args.url)
    transcriber.stream_audio()

if __name__ == "__main__":
    main()
