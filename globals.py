"""Model settings and paths, resolved relative to this repository."""
import os
from pathlib import Path

DESIRED_FEATURES = ['C', 'H', 'E']
MAX_LEN = 243
MIN_LEN = 0
TASK_TYPE = 'DNA'
MAX_LEN_PROTEIN = (MAX_LEN - 6) // 3
MIN_LEN_PROTEIN = MAX_LEN_PROTEIN
SEQ_LENGTH = MAX_LEN
DIM = 50
KERNEL_SIZE = 5
BATCH_SIZE = 128
N_CHAR = 5
NOISE_SHAPE = 128

ROOT = Path(__file__).resolve().parent
PATH_DATA = os.environ.get('FBGAN_DATA_PATH', str(ROOT / 'data' / '2018-06-06-ss.cleaned.csv'))
PATH_GAN = str(ROOT / 'weights' / 'gan')
PATH_G = os.path.join(PATH_GAN, 'E9_Generator')
PATH_D = os.path.join(PATH_GAN, 'E9_Discriminator')
PATH_FB = str(ROOT / 'weights' / 'feedback')
