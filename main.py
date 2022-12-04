import os
import tempfile
import threading
import numpy
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


def transcribe():
    file_path = q.get()

    audio = whisper.load_audio(file_path)
    os.remove(file_path)

    audio = whisper.pad_or_trim(audio)

    mean = numpy.mean(numpy.abs(audio))

    print("mean: ", mean)

    if (mean < 0.002):
        print("[Silent]")
        return

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    print(f'{max(probs, key=probs.get)}: {result.text}')


def gen_text():
    while True:
        transcribe()


def save_audio(data: bytes):
    _, output_path = tempfile.mkstemp(suffix=".wav")
    wf = wave.open(output_path, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
    return output_path


threading.Thread(target=gen_text, daemon=True).start()

try:
    buffer = []
    while True:
        n = 0
        while n < RATE * INTERVAL:
            data = stream.read(BUFFER_SIZE)
            buffer.append(data)
            n += len(data)

        output_path = save_audio(b"".join(buffer))

        q.put(output_path)

        buffer = []


# Stop when Ctrl + C is pressed
except KeyboardInterrupt:
    print("Stop recording")

    clean()

except:
    clean()
