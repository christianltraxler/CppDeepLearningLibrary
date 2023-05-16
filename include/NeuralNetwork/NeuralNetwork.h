//
// Created by Christian Traxler on 4/18/23.
//

#ifndef CPPDEEPLEARNING_NEURALNETWORK_H
#define CPPDEEPLEARNING_NEURALNETWORK_H

#include <vector>
#include <string>

#include "Layer.h"

class NeuralNetwork {
public:
    NeuralNetwork(int epochs);
    ~NeuralNetwork();

    void add(Layer *layer);
    std::vector<double> propagate(const std::vector<double> inputs);
    std::vector<std::vector<double>> getAllInputs(const std::vector<double> inputs);
    void fit(const std::vector<std::vector<double>> X, const std::vector<std::vector<double>> y);
    std::unordered_map<std::string, double> evaluate(const std::vector<std::vector<double>> X, const std::vector<double> y);

    void printWeights();

private:
    int _num_epochs;
    std::vector<double> (*_loss_function)(std::vector<double>, std::vector<double>);
    std::vector<double> (*_loss_prime_function)(std::vector<double>, std::vector<double>);
    std::vector<Layer *> _layers;
};


#endif //CPPDEEPLEARNING_NEURALNETWORK_H
