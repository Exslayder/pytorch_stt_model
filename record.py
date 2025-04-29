import sounddevice as sd
import wavio
import os
from config import DATA_DIR, SAMPLE_RATE, COUNT, PHRASE, DURATION_PHRASE

def warmup_microphone(fs=SAMPLE_RATE, duration=0.5):
    print("Прогрев микрофона...")
    sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Микрофон готов.")

def record_word(word, count=COUNT, duration=DURATION_PHRASE, fs=SAMPLE_RATE):
    path = f"{DATA_DIR}/{word}"
    os.makedirs(path, exist_ok=True)

    print(f"\nЗапись слова: {word}")
    warmup_microphone(fs)

    input("[0/{}] Нажми Enter и проговори слово для тестовой записи...".format(count))
    sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Тестовая запись завершена (не сохраняется).")

    for i in range(count):
        input(f"[{i+1}/{count}] Нажми Enter и произнеси слово '{word}'...")
        print("Используется устройство ввода:", sd.default.device)
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        filename = f"{path}/{word}_{i+1}.wav" 
        wavio.write(filename, audio, fs, sampwidth=2)
        print(f"Сохранено: {filename}")

if __name__ == "__main__":
    for word in PHRASE:
        record_word(word)
