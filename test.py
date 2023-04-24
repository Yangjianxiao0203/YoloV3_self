import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

from nets.darknet import darknet53
model=darknet53()
print(model.layers_out_filters)