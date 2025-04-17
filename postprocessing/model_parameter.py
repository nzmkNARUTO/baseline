import torch
from ptflops import get_model_complexity_info
from utils.models import ClassificationModel, MNISTCNN, CNN, generate_resnet

with torch.cuda.device(0):
    net = MNISTCNN(num_classes=26, in_channels=1)
    macs, params = get_model_complexity_info(
        model=net,
        input_res=(1, 28, 28),
        as_strings=True,
        backend="pytorch",
        print_per_layer_stat=False,
        verbose=False,
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
