import torch
import torchaudio.transforms as T
import torch.nn.functional as F
import sounddevice as sd
import numpy as np
import joblib
from train import SimpleSTT
from config import SAMPLE_RATE, N_MFCC, ISO_DIR, MODEL_DIR

def load_iso_models(words, tpl):
    return {w: joblib.load(tpl.format(w)) for w in words}


def recognize(model, transform, words, iso_models,
              threshold=0.9, duration=1.5, fs=SAMPLE_RATE, silence_rms=0.01):
    print("Говорите…")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    audio_np = audio.flatten()
    rms = np.sqrt((audio_np ** 2).mean())
    if rms < silence_rms:
        print("Не понимаю (тишина, RMS={:.4f})".format(rms))
        return

    waveform = torch.from_numpy(audio.T).float()
    features = transform(waveform).unsqueeze(0)

    with torch.no_grad():
        logits, emb = model(features)
        probs = F.softmax(logits, dim=1)
        max_prob, pred = torch.max(probs, dim=1)
        max_prob = max_prob.item()
        pred = pred.item()
        emb_np = emb.squeeze(0).cpu().numpy().reshape(1, -1)

    predicted_word = words[pred]
    iso_model = iso_models.get(predicted_word)

    if iso_model is None:
        print(f"Не понимаю (нет модели для '{predicted_word}')")
        return

    iso_decision = iso_model.predict(emb_np)[0]

    if iso_decision == 1 and max_prob >= threshold:
        print(
            f"Распознано: {predicted_word} (уверенность {max_prob:.2f}, RMS={rms:.4f})")
    else:
        print(
            f"Не понимаю (iso_decision={iso_decision}, prob={max_prob:.2f}, RMS={rms:.4f})")


if __name__ == "__main__":
    CKPT = f"{MODEL_DIR}/stt_model.pth"
    ckpt = torch.load(CKPT)
    state = ckpt["model_state_dict"]
    flat_size = ckpt["flattened_size"]
    words = ckpt["words"]

    transform = T.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, log_mels=True)
    model = SimpleSTT(flat_size, num_classes=len(words))
    model.load_state_dict(state)
    model.eval()

    iso_models = load_iso_models(words, f"{ISO_DIR}/iso_{{}}.joblib")

    print("Нажмите Enter для распознавания…")
    while True:
        input()
        recognize(model, transform, words, iso_models, threshold=0.95)
