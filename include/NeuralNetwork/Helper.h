#ifndef CPPDEEPLEARNING_HELPER_H
#define CPPDEEPLEARNING_HELPER_H

#include <vector>
#include <unordered_map>

typedef double (*afunc_ptr)(double);

enum ActivationFunction {
    LINEAR,
    SIGMOID,
    RELU
};

double linear(double num) {
    return num;
}

double linear_prime(double num) {
    return 1;
}

double sigmoid(double num) {
    return 1 / (1 + exp(-num));
}

double sigmoid_prime(double num) {
    return num * (1.0 - num);
}

double relu(double num) {
    num *= (num != 0);
    return num;
}

double relu_prime(double num) {
    return (num > 0);
}


std::pair<afunc_ptr, afunc_ptr> handleActivationFunction(ActivationFunction activation_function) {
    afunc_ptr forward, backward;
    switch (activation_function) {
        case (SIGMOID):
            forward = sigmoid;
            backward = sigmoid_prime;
        case (RELU):
            forward = relu;
            backward = relu_prime;
        default:
            forward = linear;
            backward = linear_prime;
    }

}

#endif //CPPDEEPLEARNING_HELPER_H