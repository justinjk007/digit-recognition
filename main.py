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
    # load data
    _input = torch.tensor(data.training_input, dtype=torch.float)
    _output = torch.tensor(data.training_expected_output, dtype=torch.float)
    _inputPredicted = torch.tensor(data.digitSix.flatten(), dtype=torch.float)
    _inputPredicted1 = torch.tensor(data.digitEight.flatten(), dtype=torch.float)
    _inputPredicted2 = torch.tensor(data.digitNine.flatten(), dtype=torch.float)
    _inputPredicted3 = torch.tensor(data.digitZero.flatten(), dtype=torch.float)

    ANN = NeuralNetwork(i=45, o=10, h=5)  # input,output,hidden layer size
    # weight training
    for i in range(1000):
        # mean sum squared error
        print("#" + str(i) + " error: " +
              str(torch.mean((_output - ANN(_input))**2).detach().item()))
        ANN.train(_input, _output)

    # torch.save(ANN, "algo1.weights")
    # ANN = torch.load("algo1.weights")
    print("Predicted data based on trained weights: ")
    print("Output: \n", spit_out_digit_from_output(ANN.predict(_inputPredicted)))

    print("Predicted data based on trained weights: ")
    print("Output: \n", spit_out_digit_from_output(ANN.predict(_inputPredicted1)))

    print("Predicted data based on trained weights: ")
    print("Output: \n", spit_out_digit_from_output(ANN.predict(_inputPredicted2)))

    print("Predicted data based on trained weights: ")
    print("Output: \n", spit_out_digit_from_output(ANN.predict(_inputPredicted3)))


if __name__ == "__main__":
    main()
