#ifndef CPPDEEPLEARNING_HELPER_H
#define CPPDEEPLEARNING_HELPER_H

#include "NeuralNetwork/NeuralNetwork.h"
#include "NeuralNetwork/Helper.h"



double linear(double num) {
    return num;
}

double linear_prime(double num) {
    return 1;
}

double sigmoid(double num) {
    return 1 / (1 + exp(num * -1));
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


func handleForwardActivationFunction(int activation_function) {
    switch (activation_function) {
        case (SIGMOID):
            return sigmoid;
        case (RELU):
            return relu;
        default:
            return linear;
    }

}

func handleBackwardActivationFunction(int activation_function) {
    switch (activation_function) {
        case (SIGMOID):
            return sigmoid_prime;
        case (RELU):
            return relu_prime;
        default:
            return linear_prime;
    }

}

#endif //CPPDEEPLEARNING_HELPER_H