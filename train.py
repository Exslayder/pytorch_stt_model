import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import WordsDataset
import torchaudio.transforms as T
from sklearn.ensemble import IsolationForest
import joblib
from config import SAMPLE_RATE, N_MFCC, ISO_DIR, MODEL_DIR

class SimpleSTT(nn.Module):
    def __init__(self, flattened_size, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        emb = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits, emb


if __name__ == "__main__":
    words = ["hello", "yes", "spotify"]
    transform = T.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, log_mels=True)
    dataset = WordsDataset(words, transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    sample, _ = dataset[0]
    sample = sample.unsqueeze(0)    
    dummy_conv = nn.Sequential(*list(SimpleSTT(0, 0).conv)) 
    out = dummy_conv(sample)
    flattened_size = out.view(1, -1).shape[1]

    model = SimpleSTT(flattened_size, num_classes=len(words))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        for x, y in loader:
            logits, emb = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Эпоха {epoch+1}, loss: {total_loss:.4f}")

    model.eval()
    all_embs = {w: [] for w in words}
    with torch.no_grad():
        for x, y in loader:
            _, emb = model(x)
            for i, label in enumerate(y):
                all_embs[words[label]].append(emb[i].cpu())

    centroids = {}
    max_dists = []
    for w, embs in all_embs.items():
        embs = torch.stack(embs) 
        centroid = embs.mean(dim=0)
        dists = torch.norm(embs - centroid.unsqueeze(0), dim=1)
        centroids[w] = centroid
        max_dists.append(dists.max().item())

    threshold = max(max_dists)

    torch.save({
        'model_state_dict': model.state_dict(),
        'flattened_size': flattened_size,
        'words': words,
        'centroids': centroids,
        'threshold': threshold
    }, f"{MODEL_DIR}/stt_model.pth")

    iso_models = {}
    for word, embs in all_embs.items():
        X = torch.stack(embs).numpy()
        iso = IsolationForest(contamination=0.1, random_state=0).fit(X)
        iso_models[word] = iso
        joblib.dump(iso, f"{ISO_DIR}/iso_{word}.joblib")
