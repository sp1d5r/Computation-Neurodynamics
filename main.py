from NetworkScience.network.Network import Network
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    largest_number_of_nodes = 200
    smallest_number_of_nodes = 100
    largest_neighbourhood_size = 100
    largest_p = 100

    small_world_index = [
        [
            [0 for _ in range(smallest_number_of_nodes, largest_number_of_nodes, 5)]
            for _ in range(1, largest_neighbourhood_size)
        ]
        for _ in range(0, largest_p, 5)
    ]

    for nodes in range(smallest_number_of_nodes, largest_number_of_nodes, 5):
        for neighbourhood_size in range(5, largest_neighbourhood_size, 5):
            for probability in range(0, largest_p, 5):
                prob = probability/100
                network = Network(nodes=[], edges=[])
                print(nodes, neighbourhood_size, prob)
                network.generate_watts_strogatz(nodes=nodes, k=neighbourhood_size, p=prob)
                small_world_index_of_network = network.calculate_small_world_index()
                print(small_world_index_of_network)
                small_world_index[probability//5][(neighbourhood_size - 5)//5][(nodes - smallest_number_of_nodes)//5] = small_world_index_of_network
                network.save_network_to_file(f"./assets/network{nodes}-{neighbourhood_size}-{probability}.png")
        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Prepare data
        X = np.arange(smallest_number_of_nodes, largest_number_of_nodes, 5)
        Y = np.arange(0, largest_neighbourhood_size, 1)
        X, Y = np.meshgrid(X, Y)

        Z = np.array(small_world_index[0])  # Replace as needed

        # Plotting the surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')

        # Add color bar
        fig.colorbar(surf)

        # Labels and title
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Neighbourhood Size')
        ax.set_zlabel('Small World Index')
        ax.set_title(f'3D plot of Small World Index for {nodes} Nodes')

        # Save figure
        plt.savefig(f"./assets/3D_Small_World_Index_{nodes}.png")

        # Clear the current figure to avoid overlap in the next iteration
        plt.clf()
    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data
    X = np.arange(smallest_number_of_nodes, largest_number_of_nodes, 5)
    Y = np.arange(0, largest_neighbourhood_size, 1)
    X, Y = np.meshgrid(X, Y)

    # Let's assume Z is your small_world_index for some value of probability
    # Replace this line with how you obtain Z from small_world_index
    Z = np.array(small_world_index[0])  # This should be a 2D array

    # Plotting the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')

    # Add color bar
    fig.colorbar(surf)

    # Labels and title
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Neighbourhood Size')
    ax.set_zlabel('Small World Index')
    ax.set_title('3D plot of Small World Index')

    # Save figure
    plt.savefig("3D_Small_World_Index.png")

    # Show plot
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
