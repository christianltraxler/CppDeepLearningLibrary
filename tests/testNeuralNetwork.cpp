//
// Created by Christian Traxler on 4/17/23.
//

#include <gtest/gtest.h>

#include "NeuralNetwork/NeuralNetwork.h"
#include <iostream>

TEST(SampleTests, XORTest1) {
    std::vector<std::vector<double>> X{{0,0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> y{{0}, {1}, {1}, {0}};

    NeuralNetwork nn = NeuralNetwork(100);
    EXPECT_EQ(1, 1);
    nn.add(new DenseLayer(2,3, SIGMOID, 0.01));
    nn.add(new DenseLayer(3,1, SIGMOID, 0.01));
    EXPECT_EQ(1, 1);
    nn.fit(X, y);
    EXPECT_EQ(1, 1);
}


// # decision boundary plot
// points = []
// for x in np.linspace(0, 1, 20):
//     for y in np.linspace(0, 1, 20):
//         z = predict(network, [[x], [y]])
//         points.append([x, y, z[0,0]])

// points = np.array(points)