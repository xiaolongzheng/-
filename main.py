from sys import argv

import numpy as np, os

from benchmarks.tsp.utils import read_dataset, normalize_2d
from som.neuron import get_neighborhood, get_route
from som.distance import select_closest, euclidean_distance, route_distance
from benchmarks.tsp.plot import plot_network, plot_route

def main():


    problem = read_dataset("uy734")

    route = som(problem, 200000)

    problem = problem["node_coords"][route]

    distance = route_distance(problem)

    print('Route found of length {}'.format(distance))


def som(problem, iterations, learning_rate=0.8, rnd=np.random):
    """Solve the TSP using a Self-Organizing Map."""

    # The population size is 8 times the number of cities
    n = problem["num_nodes"] * 8
    cities = normalize_2d(problem["node_coords"])

    # Generate an adequate network of neurons:
    network = rnd.rand(n, 2)*2-1
    network[:, 1] = np.sqrt(1-(network[:, 0])**2) * (-1)**np.random.randint(2, size=network.shape[0])
    network = network/2+0.5
    # A = np.random.randint(2, size=network.shape[0])
    # A = (-1)**np.random.randint(2, size=network.shape[0])
    # A = network[:,0]**2 + network[:,1]**2
    print('Network of {} neurons created. Starting the iterations:'.format(n))

    plot_network(cities, network, name='diagrams/{}.png'.format("0000o"))

    for i in range(iterations):
        if not i % 100:
            print('\t> Iteration {}/{}'.format(i, iterations), end="\r")
        # Choose a random city
        city = cities[rnd.randint(cities.shape[0]), :]
        winner_idx = select_closest(network, city)  # winner_idx in the network
        # Generate a filter that applies changes to the winner's gaussian
        gaussian = get_neighborhood(winner_idx, n//10, network.shape[0])
        # Update the network's weights (closer to the city)
        network += gaussian[:,np.newaxis] * learning_rate * (city - network)
        # Decay the variables
        learning_rate = learning_rate * 0.99997
        n = n * 0.9997

        # Check for plotting interval
        if not i % 100:
            plot_network(cities, network, name='diagrams/{:05d}.png'.format(i))

        # Check if any parameter has completely decayed.
        if n < 1:
            print('Radius has completely decayed, finishing execution',
            'at {} iterations'.format(i))
            break
        if learning_rate < 0.001:
            print('Learning rate has completely decayed, finishing execution',
            'at {} iterations'.format(i))
            break
    else:
        print('Completed {} iterations.'.format(iterations))

    plot_network(cities, network, name='diagrams/final.png')

    route = get_route(cities, network)
    plot_route(cities, route, 'diagrams/route.png')
    return route

if __name__ == '__main__':
    main()
