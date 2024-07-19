import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import model 

Model, parameter_count = model.Model_loader()

device = 'cuda' if torch.cuda.is_available() else "cpu"
Model = Model.to(device)

