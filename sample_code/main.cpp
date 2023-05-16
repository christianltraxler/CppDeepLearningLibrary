#include "NeuralNetwork/NeuralNetwork.h"
#include <iostream>
int main(int argc, char *argv[])
{
    std::vector<std::vector<double>> X{{0,0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> y{{0}, {1}, {1}, {0}};

    NeuralNetwork nn = NeuralNetwork(5);
    nn.add(new DenseLayer(2,3, SIGMOID, 0.01));
    nn.add(new DenseLayer(3,1, SIGMOID, 0.01));
    nn.fit(X, y);
    std::cout << nn.propagate({0,0})[0] << std::endl;
    std::cout << nn.propagate({0,1})[0] << std::endl;
    std::cout << nn.propagate({1,0})[0] << std::endl;
    std::cout << nn.propagate({1,1})[0] << std::endl;
    return 0;
}
