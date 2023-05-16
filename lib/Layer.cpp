//
// Created by Christian Traxler on 4/18/23.
//

#include "NeuralNetwork/Layer.h"
#include <iostream>


Layer::Layer(int inputs, int outputs, int activation_function, double learning_rate) {
    _num_inputs = inputs;
    _num_outputs = outputs;
    _learning_rate = learning_rate;
    _forward_activation = handleForwardActivationFunction(activation_function);
    _backward_activation = handleBackwardActivationFunction(activation_function);
}

Layer::~Layer() {}

std::vector<double> Layer::forwardPropagate(const std::vector<double> inputs) {
    return std::vector<double>();
}

std::vector<double> Layer::backPropagate(const std::vector<double> inputs, std::vector<double> gradients) {
    return std::vector<double>();
}



DenseLayer::DenseLayer(int inputs, int outputs, int activation_function, double learning_rate) : Layer(inputs, outputs, activation_function, learning_rate) {
    _weights = std::vector<std::vector<double>>(outputs, std::vector<double>(inputs));
    _biases = std::vector<double>(outputs);
    for(int i = 0; i < _num_outputs; i++) {
        _biases[i] = (double) (rand() % 1);
        for (int j = 0; j < _num_inputs; j++) {
            _weights[i][j] = (double) (rand() % 1);
        }
    }
}

DenseLayer::~DenseLayer() {}

std::vector<double> DenseLayer::forwardPropagate(const std::vector<double> inputs) {
    assert(inputs.size() == _num_inputs);
    assert(_weights.size() == _num_outputs);
    assert(_weights[0].size() == _num_inputs);
    std::vector<double> outputs(_num_outputs);
    for (int i = 0; i < _num_outputs; i++) {
        outputs[i] = _biases[i];
        for (int j = 0; j < _num_inputs; j++) {
            outputs[i] += inputs[j] * _weights[i][j];
        }
        outputs[i] = _forward_activation(outputs[i]);
    }
    return outputs;
}

std::vector<double> DenseLayer::backPropagate(const std::vector<double> inputs, std::vector<double> output_gradients) {
    std::vector<double> input_gradients(_num_inputs, 0);

    for (int i = 0; i < _num_outputs; i++) {
        // Update gradients using derivative of activation function
        output_gradients[i] = _backward_activation(output_gradients[i]);

        // Update the biases dE/dB = dE/dY * dY/dB = dE/dY
        _biases[i] -= _learning_rate * output_gradients[i];
        for (int j = 0; j < _num_inputs; j++) {
            // Set the input gradient for the next layer in the backward pass
            //  - Must be done before updating weights
            input_gradients[j] += _weights[i][j] * output_gradients[i];

            // Update the weights dE/dW = dE/dY * dY/dW
            _weights[i][j] -= _learning_rate * output_gradients[i] * inputs[j];
        }
    }
    return input_gradients;
}

void Layer::printWeights() {
    for (int i = 0; i < _weights.size(); i++)
    {
        for (int j = 0; j < _weights[i].size(); j++)
        {
            std::cout << _weights[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
