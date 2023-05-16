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
    NeuralNetwork();
    ~NeuralNetwork();

    void add(Layer *layer);
    std::vector<double> propagate(const std::vector<double> inputs);
    void fit(const std::vector<std::vector<double>> X, const std::vector<double> y);
    std::unordered_map<std::string, double> evaluate(const std::vector<std::vector<double>> X, const std::vector<double> y);

private:
    std::vector<Layer *> _layers;
};


#endif //CPPDEEPLEARNING_NEURALNETWORK_H
