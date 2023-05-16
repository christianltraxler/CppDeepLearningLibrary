//
// Created by Christian Traxler on 4/18/23.
//

#ifndef CPPDEEPLEARNINGTESTS_LAYER_H
#define CPPDEEPLEARNINGTESTS_LAYER_H

#include <vector>
#include <unordered_map>


enum ActivationFunction {
    LINEAR,
    SIGMOID,
    RELU
};


class Layer {
public:
    Layer(int inputs, int outputs, ActivationFunction a);
    ~Layer();

    virtual std::vector<double> forwardPropagate(std::vector<double> inputs);
    virtual void backPropagate(std::vector<double> inputs, std::vector<double> gradients);

private:
    int _num_inputs;
    int _num_outputs;
    ActivationFunction _activation_function;
    std::vector<double> _weights;
    std::vector<double> _biases;
};

class DenseLayer : private Layer {
public:
    DenseLayer(int inputs, int outputs, ActivationFunction a);
    ~DenseLayer();

    std::vector<double> forwardPropagate(std::vector<double> inputs);
    void backPropagate(std::vector<double> inputs, std::vector<double> gradients);
};

#endif //CPPDEEPLEARNINGTESTS_LAYER_H
