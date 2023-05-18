#ifndef CPPDEEPLEARNING_HELPER_H
#define CPPDEEPLEARNING_HELPER_H

#include <cmath>

#include "Helper.h"

typedef double (*func)(double);

enum ActivationFunction {
    LINEAR,
    SIGMOID,
    RELU,
    TANH,
};

double linear_func(double num);
double linear_prime(double num);

double sigmoid(double num);
double sigmoid_prime(double num);

double relu(double num) ;
double relu_prime(double num);

double dTanh(double num);

func handleForwardActivationFunction(int activation_function);
func handleBackwardActivationFunction(int activation_function);


#endif //CPPDEEPLEARNING_HELPER_H