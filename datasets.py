import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import glob
from config import SAMPLE_RATE, N_MFCC, DATA_DIR

class WordsDataset(Dataset):
    def __init__(self, words, transform=None):
        self.files = []
        self.labels = []
        self.words = words
        self.transform = transform or T.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC)

        for idx, word in enumerate(words):
            path = f"{DATA_DIR}/{word}"
            for f in glob.glob(f"{path}/*.wav"):
                self.files.append(f)
                self.labels.append(idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        waveform, sr = torchaudio.load(self.files[index])
        mfcc = self.transform(waveform)
        return mfcc, self.labels[index]
