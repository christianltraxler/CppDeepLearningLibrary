#include "NeuralNetwork/Helper.h"

double linear(double num) {
    return num;
}

double dLinear(double num) {
    return 1;
}

double sigmoid(double num) {
    return 1 / (1 + exp(num * -1));
}

double dSigmoid(double num) {
    return sigmoid(num) * (1.0 - sigmoid(num));
}

double relu(double num) {
    num *= (num != 0);
    return num;
}

double dRelu(double num) {
    return (num > 0);
}

double dTanh(const double x) {
    double th = tanh(x); // dTanh(x) = sech^2(x) = 1 - tanh^2(x)
    return 1.0 - th*th; 
}

func handleForwardActivationFunction(int activation_function) {
    switch (activation_function) {
        case (SIGMOID):
            return sigmoid;
        case (RELU):
            return relu;
        case (TANH):
            return tanh;
        default:
            return linear;
    }

}

func handleBackwardActivationFunction(int activation_function) {
    switch (activation_function) {
        case (SIGMOID):
            return dSigmoid;
        case (RELU):
            return dRelu;
        case (TANH):
            return dTanh;
        default:
            return dLinear;
    }

}