#include <iostream>

#include "NeuralNetwork/NeuralNetwork.h"


void sample_xor_example();


int main(int argc, char *argv[])
{
    srand((unsigned int)time(NULL));
    
    sample_xor_example();

    return 0;
}


void sample_xor_example()
{
    std::vector<std::vector<double>> X{{0,0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> y{{0}, {1}, {1}, {0}};

    NeuralNetwork nn = NeuralNetwork(10000);
    nn.add(new DenseLayer(2,8, TANH, 0.1));
    nn.add(new DenseLayer(8,8, SIGMOID, 0.1));
    nn.add(new DenseLayer(8,1, TANH, 0.1));

    nn.fit(X, y);
    
    for (int i = 0; i < X.size(); i++) {
        std::cout << "Inputs: " << X[i][0] << ", " << X[i][1] << ". ";
        std::cout << "Outputs: " << nn.propagate(X[i])[0] << ". ";
        std::cout << "Expected: " << y[i][0] << "." << std::endl;
    }
}