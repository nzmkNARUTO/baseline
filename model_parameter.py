import torch
from ptflops import get_model_complexity_info
from utils.models import ClassificationModel
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms

with torch.cuda.device(0):
    net = ClassificationModel(28 * 28, 10)
    macs, params = get_model_complexity_info(
        net,
        (28 * 28,),
        as_strings=True,
        backend="pytorch",
        print_per_layer_stat=False,
        verbose=False,
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
