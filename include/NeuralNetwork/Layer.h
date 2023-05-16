//
// Created by Christian Traxler on 4/18/23.
//

#ifndef CPPDEEPLEARNING_LAYER_H
#define CPPDEEPLEARNING_LAYER_H

#include <vector>
#include <unordered_map>

#include "Helper.h"

typedef double (*func)(double);
enum ActivationFunction {
    LINEAR,
    SIGMOID,
    RELU
};


class Layer {
public:
    Layer(int inputs, int outputs, int activation_function, double learning_rate);
    ~Layer();

    virtual std::vector<double> forwardPropagate(std::vector<double> inputs);
    virtual std::vector<double> backPropagate(const std::vector<double> inputs, const std::vector<double> gradients);
    virtual void printWeights();

protected:
    int _num_inputs;
    int _num_outputs;
    double _learning_rate;

    double (*_forward_activation)(double);
    double (*_backward_activation)(double);

    std::vector<std::vector<double>> _weights;
    std::vector<double> _biases;
};

class DenseLayer : public Layer {
public:
    DenseLayer(int inputs, int outputs, int activation_function, double learning_rate);
    ~DenseLayer();

    std::vector<double> forwardPropagate(std::vector<double> inputs);
    std::vector<double> backPropagate(const std::vector<double> inputs, const std::vector<double> output_gradients);
};

#endif //CPPDEEPLEARNING_LAYER_H
