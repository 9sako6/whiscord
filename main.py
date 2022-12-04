import os
import tempfile
import threading
import pyaudio
import wave
import whisper
import argparse
import queue

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
INTERVAL = 10
BUFFER_SIZE = 4096

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='base')
args = parser.parse_args()

print('Loading model...')
model = whisper.load_model(args.model)

pa = pyaudio.PyAudio()

stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 frames_per_buffer=BUFFER_SIZE
                 )

print("Start recording")

q = queue.Queue()


def clean():
    stream.stop_stream()
    stream.close()
    pa.terminate()

    while not (q.empty()):
        file_path = q.get()
        os.remove(file_path)


def gen_text():
    while True:
        file_path = q.get()

        audio = whisper.load_audio(file_path)
        os.remove(file_path)

        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        # detect the spoken language
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        # decode the audio
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        print(f'{max(probs, key=probs.get)}: {result.text}')


threading.Thread(target=gen_text, daemon=True).start()

try:
    while True:
        frames = []
        n = 0
        while n < RATE * INTERVAL:
            data = stream.read(BUFFER_SIZE)
            frames.append(data)
            n += len(data)

        _, output_path = tempfile.mkstemp(suffix=".wav")
        wf = wave.open(output_path, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        wf.close()

        q.put(output_path)


# Stop when Ctrl + C is pressed
except KeyboardInterrupt:
    print("Stop recording")

    clean()

except:
    clean()
