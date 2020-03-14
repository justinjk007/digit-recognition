import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, i, o, h):
        super().__init__() # recommended by pytorch
        self.input_num = i
        self.output_num = o
        self.hidden_num = h
        # returns a tensor with random values, weights applicable for input
        # layer and hidden layer
        self.w1 = torch.randn(self.input_num, self.hidden_num)
        self.w2 = torch.randn(self.hidden_num, self.output_num)

    def forward(self, x):
        self.z = torch.matmul(x, self.w1)  # matrix multiplication
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = torch.matmul(self.z2, self.w2)
        output = self.sigmoid(self.z3)  # final activation function
        return output

    def sigmoid(self, x):
        # x can be a tensor of any size
        return 1 / (1 + torch.exp(-x))

    def sigmoidPrime(self, x):
        # find the derivative the tensor x
        return x * (1 - x)

    def backward(self, x, y, o):
        self.o_error = y - o  # error in output
        # derivative of sig to error
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.w2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.w1 += torch.matmul(torch.t(x), self.z2_delta)
        self.w2 += torch.matmul(torch.t(self.z2), self.o_delta)

    def train(self, x, y):
        # forward + backward pass for training
        o = self.forward(x)
        self.backward(x, y, o)
