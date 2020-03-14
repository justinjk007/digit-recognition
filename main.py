import torch
import torch.nn as nn
from neural_net import NeuralNetwork

x = torch.tensor(([2, 9], [1, 5], [3, 6]), dtype=torch.float) # 3 X 2 tensor
y = torch.tensor(([92], [100], [89]), dtype=torch.float) # 3 X 1 tensor
xPredicted = torch.tensor(([4, 8]), dtype=torch.float) # 1 X 2 tensor

# scale units
x_max, _ = torch.max(x, 0)
xPredicted_max, _ = torch.max(xPredicted, 0)

x = torch.div(x, x_max)
xPredicted = torch.div(xPredicted, xPredicted_max)
y = y / 100  # max test score is 100, so this makes everything out of 1

ANN = NeuralNetwork(2,1,3)
for i in range(1000):  # trains the NN 1,000 times
    print ("#" + str(i) + " Loss: " + str(torch.mean((y - ANN(x))**2).detach().item()))  # mean sum squared loss
    ANN.train(x, y)

torch.save(ANN, "algo1.weights")
# torch.load("algo1.weights")
print("Predicted data based on trained weights: ")
print("Input (scaled): \n" + str(xPredicted))
print("Output: \n" + str(ANN.forward(xPredicted)))
