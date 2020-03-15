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


def test_trained_network(ANN):
    print("Trained network being tested...\n")
    error_count = 0
    for i in range(len(data.testing_input)):
        _input = torch.tensor(data.testing_input[i], dtype=torch.float)
        exp_output = data.testing_output[i]
        print("Expected output  : ", exp_output)
        _output = ANN.predict(_input)
        print("Predicted output : ", spit_out_digit_from_output(_output),"\n")
        if(spit_out_digit_from_output(_output) != exp_output):
            error_count+=1
    print("Error count after testing ", len(data.testing_input), "inputs: ", error_count)


def main():
    # load data
    _input = torch.tensor(data.training_input, dtype=torch.float)
    _output = torch.tensor(data.training_expected_output, dtype=torch.float)

    ANN = NeuralNetwork(i=45, o=10, h=5)  # input,output,hidden layer size
    # weight training
    for i in range(70000):
        # mean sum squared error
        print("#" + str(i) + " error: " +
              str(torch.mean((_output - ANN(_input))**2).detach().item()))
        ANN.train(_input, _output)

    torch.save(ANN, "algo1.weights")
    # ANN = torch.load("algo1.weights")
    test_trained_network(ANN)


if __name__ == "__main__":
    main()
