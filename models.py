# models used in defect detection
# author: Zhaonan Li, zli@brandeis.edu
# created at: 4/17/2020
import torch
import torch.nn as nn
from grayscale_resnet import resnet34, resnet18


# integrator class
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# load hard scanner, uniform scanner and integrator to device
def load_models(path_hard, path_uniform, path_int, device):
    num_classes = 5

    # initialize models
    model_hard = resnet18()
    num_ftrs = model_hard.fc.in_features
    model_hard.fc = nn.Linear(num_ftrs, num_classes)

    model_uniform = resnet34()
    num_ftrs = model_uniform.fc.in_features
    model_uniform.fc = nn.Linear(num_ftrs, num_classes)

    model_int = Net(10, 100, 5)

    # load state_dict from file
    hard_dict = torch.load(path_hard, map_location=device)
    uniform_dict = torch.load(path_uniform, map_location=device)
    int_dict = torch.load(path_int, map_location=device)

    model_hard.load_state_dict(hard_dict)
    model_uniform.load_state_dict(uniform_dict)
    model_int.load_state_dict(int_dict)

    model_hard.train(False)
    model_uniform.train(False)
    model_int.train(False)

    model_hard.to(device)
    model_uniform.to(device)
    model_int.to(device)

    return model_hard, model_uniform, model_int
