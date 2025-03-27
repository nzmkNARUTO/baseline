import torch
from ptflops import get_model_complexity_info
from utils.models import ClassificationModel
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from preprocessing.baselines_dataloader import load_data

load_data("EMNIST")
