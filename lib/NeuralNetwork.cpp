//
// Created by Christian Traxler on 4/18/23.
//

#include "NeuralNetwork/NeuralNetwork.h"
#include <iostream>

double mse(std::vector<double> y_true, std::vector<double> y_pred) {
    assert(y_true.size() == y_pred.size());

    double result;
    for (int i = 0; i < y_true.size(); i++) {
        result += pow(y_true[i] - y_pred[i], 2);
    }
    result /= y_true.size();
    return result;
}

std::vector<double> mse_prime(std::vector<double> y_true, std::vector<double> y_pred) {
    assert(y_true.size() == y_pred.size());

    double result = 0;
    for (int i = 0; i < y_true.size(); i++) {
        result += (y_true[i] - y_pred[i]);
    }
    result *= 2;
    result /= y_true.size();

    return std::vector<double>(y_pred.size(), result);
}

NeuralNetwork::NeuralNetwork(int epochs) {
    _num_epochs = epochs;
    _loss_function = mse;
    _loss_prime_function = mse_prime;
}

NeuralNetwork::~NeuralNetwork() {

}

void NeuralNetwork::add(Layer *l) {
    _layers.push_back(l);
}

std::vector<double> NeuralNetwork::propagate(const std::vector<double> inputs) {
    
    std::vector<double> outputs = inputs;
    for (auto layer : _layers) {
        outputs = layer->forwardPropagate(outputs);
    }
    return outputs;
}

std::vector<std::vector<double>> NeuralNetwork::getAllInputs(const std::vector<double> inputs) {
    
    std::vector<std::vector<double>> results{inputs};
    for (auto layer : _layers) {
        results.push_back(layer->forwardPropagate(results.back()));
    }
    return results;
}

void NeuralNetwork::fit(const std::vector<std::vector<double>> X, const std::vector<std::vector<double>> y) {
    assert(X.size() == y.size());
    for (int e = 0; e < _num_epochs; e++) {
        double error = 0;
        for (int i = 0; i < X.size(); i++) {
            std::vector<std::vector<double>> inputs = getAllInputs(X[i]);
            error += _loss_function(y[i], inputs.back());
            std::vector<double> gradient = _loss_prime_function(y[i], inputs.back());
            for (int l = _layers.size()-1; l >= 0; l--) {
                gradient = _layers[l]->backPropagate(inputs[l], gradient);
            }
        }
        std::cout << "Epoch " << e+1 << " with error " << error << std::endl;
    }
}


void NeuralNetwork::printWeights() {
    for (auto layer : _layers) {
        layer->printWeights();
        std::cout << std::endl;
    }
}

std::unordered_map<std::string, double> NeuralNetwork::evaluate(std::vector<std::vector<double>> X, std::vector<double> y) {
    return std::unordered_map<std::string, double>();
}