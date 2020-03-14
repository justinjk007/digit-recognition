import torch
import torch.nn as nn
from neural_net import NeuralNetwork
import data


def spit_out_digit_from_output(output):
    # here _ is the index of the max value. For our output the
    # expected output neuron should be activated the most so the
    # maximum digits index numner should be the actual digit
    max_val, _ = torch.max(output, 0)
    return _.item()  # return the index number


def main():
    x = torch.tensor(data.training_input, dtype=torch.float)
    y = torch.tensor(data.training_expected_output, dtype=torch.float)
    xPredicted = torch.tensor(data.digitSix.flatten(), dtype=torch.float)
    xPredicted1 = torch.tensor(data.digitEight.flatten(), dtype=torch.float)
    xPredicted2 = torch.tensor(data.digitNine.flatten(), dtype=torch.float)
    xPredicted3 = torch.tensor(data.digitZero.flatten(), dtype=torch.float)

    ANN = NeuralNetwork(45, 10, 5)
    for i in range(1000):  # trains the NN many times
        # mean sum squared error
        print("#" + str(i) + " error: " +
              str(torch.mean((y - ANN(x))**2).detach().item()))
        ANN.train(x, y)

    # torch.save(ANN, "algo1.weights")
    # ANN = torch.load("algo1.weights")
    print("Predicted data based on trained weights: ")
    print("Output: \n", spit_out_digit_from_output(ANN.predict(xPredicted)))

    print("Predicted data based on trained weights: ")
    print("Output: \n", spit_out_digit_from_output(ANN.predict(xPredicted1)))

    print("Predicted data based on trained weights: ")
    print("Output: \n", spit_out_digit_from_output(ANN.predict(xPredicted2)))

    print("Predicted data based on trained weights: ")
    print("Output: \n", spit_out_digit_from_output(ANN.predict(xPredicted3)))


if __name__ == "__main__":
    main()
