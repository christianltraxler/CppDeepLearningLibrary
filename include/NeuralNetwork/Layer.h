//
// Created by Christian Traxler on 4/18/23.
//

#ifndef CPPDEEPLEARNING_LAYER_H
#define CPPDEEPLEARNING_LAYER_H

#include <vector>
#include <unordered_map>

#include "Helper.h"


enum ActivationFunction {
    LINEAR,
    SIGMOID,
    RELU
};


class Layer {
public:
    Layer(int inputs, int outputs, ActivationFunction activation_function, double learning_rate);
    ~Layer();

    virtual std::vector<double> forwardPropagate(std::vector<double> inputs);
    virtual std::vector<double> backPropagate(const std::vector<double> inputs, const std::vector<double> gradients);

protected:
    int _num_inputs;
    int _num_outputs;
    afunc_ptr _forward_activation;
    afunc_ptr _backward_activation;
    double _learning_rate;

    std::vector<std::vector<double>> _weights;
    std::vector<double> _biases;
};

class DenseLayer : public Layer {
public:
    DenseLayer(int inputs, int outputs, ActivationFunction activation_function, double learning_rate);
    ~DenseLayer();

    std::vector<double> forwardPropagate(std::vector<double> inputs);
    std::vector<double> backPropagate(const std::vector<double> inputs, const std::vector<double> output_gradients);
};

#endif //CPPDEEPLEARNING_LAYER_H
