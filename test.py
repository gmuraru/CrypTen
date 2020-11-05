import crypten

import torch
from torchvision import models
import crypten.communicator as comm
from collections import OrderedDict


crypten.init()

def construct_private_model():
    rank = comm.get().get_rank()
    f = open("resnet/resnet18-v1-7.onnx", "rb")
    private_model = crypten.nn.from_onnx(f)
    f.close()

construct_private_model()
