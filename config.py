import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
MODEL_DIR = os.getenv("MODEL_DIR")
ISO_DIR = os.getenv("ISO_DIR")

SAMPLE_RATE = 16000
N_MFCC = 40
COUNT = 20 # Количество записей для одного слова
PHRASE = ["yes","no","hellow"] # Фразы
DURATION_PHRASE = 1.5 # Время записи