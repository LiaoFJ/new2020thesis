import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.utils as vutils
import time
import torch.nn as nn
from Final_model import Final_model

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='./checkpoint/model_final.pkl', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=16, help='Number of generated outputs')
parser.add_argument('-t', default=None, help='Number of glimpses.')
args = parser.parse_args()