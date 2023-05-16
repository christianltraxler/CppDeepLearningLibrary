//
// Created by Christian Traxler on 4/18/23.
//

#include "NeuralNetwork/Layer.h"


Layer::Layer(int inputs, int outputs, ActivationFunction activation_function, double learning_rate) {
    _num_inputs = inputs;
    _num_outputs = outputs;
    _learning_rate = learning_rate;

    std::pair<afunc_ptr, afunc_ptr> activation_functions = handleActivationFunction(activation_function);
    _forward_activation = activation_functions.first;
    _backward_activation = activation_functions.second;
}

Layer::~Layer() {}

std::vector<double> Layer::forwardPropagate(const std::vector<double> inputs) {}

std::vector<double> Layer::backPropagate(const std::vector<double> inputs, std::vector<double> gradients) {}



DenseLayer::DenseLayer(int inputs, int outputs, ActivationFunction activation_function, double learning_rate) : Layer(inputs, outputs, activation_function, learning_rate) {
    _weights = std::vector<std::vector<double>>(outputs, std::vector<double>(inputs));
    _biases = std::vector<double>(outputs);
}

DenseLayer::~DenseLayer() {}

std::vector<double> DenseLayer::forwardPropagate(const std::vector<double> inputs) {
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
