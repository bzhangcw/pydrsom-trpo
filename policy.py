import torch
import torch.nn.functional as func

class policy(self):
    def __init__(self):
        pass

    def forward(self, input):
        input = self.module(torch.Tensor(input))
        input = func.softmax(input, dim=1)

        return Categorical(input)





