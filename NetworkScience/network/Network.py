from .Edge import Edge
from .Node import Node
import matplotlib.pyplot as plt
import random
import networkx as nx
import math


class Network:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def connect_nodes(self, node_a, node_b):
        assert node_a in self.nodes
        assert node_b in self.nodes
        edge = Edge(node_a=node_a, node_b=node_b)
        node_a.add_edge(edge)
        node_b.add_edge(edge)

    def visualise_network(self):
        G = nx.Graph()

        for node in self.nodes:
            G.add_node(node.key)

        for edge in self.edges:
            G.add_edge(edge.node_a.key, edge.node_b.key)

        pos = nx.spring_layout(G)
        labels = {node.id: node.key for node in self.nodes}

        nx.draw(G, pos, labels=labels, with_labels=True, node_color="skyblue", font_weight='bold', node_size=700,
                font_size=18)
        plt.show()

    def save_network_to_file(self, filename):
        G = nx.Graph()

        for node in self.nodes:
            G.add_node(node.key)

        for edge in self.edges:
            G.add_edge(edge.node_a.key, edge.node_b.key)

        pos = nx.spring_layout(G)
        labels = {node.key: node.key for node in self.nodes}

        nx.draw(G, pos, labels=labels, with_labels=True, node_color="skyblue", font_weight='bold', node_size=700,
                font_size=18)
        plt.savefig(filename)
        plt.clf()
        plt.close()

    def generate_watts_strogatz(self, nodes=12, k=4, p=0.1):
        # Create nodes and add them to the network
        for i in range(nodes):
            new_node = Node(i)
            self.add_node(new_node)

        # Create a ring of nodes and connect each node to its k nearest neighbors
        for i in range(nodes):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % nodes
                edge = Edge(self.nodes[i], self.nodes[neighbor])
                self.nodes[i].add_edge(edge)
                self.nodes[neighbor].add_edge(edge)
                self.add_edge(edge)

        # Rewire edges with probability p
        for edge in self.edges:
            if random.random() < p:
                node_a = edge.node_a
                # Choose a new node to connect, avoiding self-loops and existing edges
                while True:
                    new_node_b = random.choice(self.nodes)
                    new_edge = Edge(node_a, new_node_b)
                    if new_node_b != node_a and new_edge not in self.edges:
                        break

                # Remove the original edge and add the new edge
                self.edges.remove(edge)
                node_a.remove_edge(edge)
                edge.node_b.remove_edge(edge)

                self.add_edge(new_edge)
                node_a.add_edge(new_edge)
                new_node_b.add_edge(new_edge)


    def shortest_path(self, start_node, end_node):
        visited = []
        queue = [(start_node, 0)] # node, depth

        while queue:
            current_node, depth = queue.pop(0)

            if current_node == end_node:
                return depth

            if current_node in visited:
                continue

            neighbours = current_node.get_neighbours()

            visited.append(current_node)

            for neighbour in neighbours:
                if neighbour not in visited:
                    queue.append((neighbour, depth + 1))

        return -1

    def average_shortest_path_length(self):
        total_length = 0
        total_pairs = 0

        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                length = self.shortest_path(self.nodes[i], self.nodes[j])
                if length is not None:  # Only count pairs that are connected
                    total_length += length
                    total_pairs += 1

        return total_length / total_pairs if total_pairs > 0 else None

    def get_clustering_coefficient(self):
        clustering_coefficient = []

        for node in self.nodes:
            neighbours = node.get_neighbours()
            actual_connections = 0
            total_connections = 0
            for neighbour in neighbours:
                for other_neighbour in neighbours:
                    if neighbour != other_neighbour:
                        if neighbour.is_connected(other_neighbour):
                            actual_connections += 1
                        total_connections += 1

            if total_connections == 0:
                continue

            clustering_coefficient.append(actual_connections / total_connections)

        return sum(clustering_coefficient) / len(clustering_coefficient)

    def get_average_degree(self):
        degrees = [node.get_degree() for node in self.nodes]
        return sum(degrees)/ len(degrees)

    def calculate_small_world_index(self):
        number_of_nodes = len(self.nodes)
        number_of_edges = len(self.edges)

        clustering_coefficient_of_random_network = number_of_edges / (math.comb(number_of_nodes, 2))
        clustering_coefficient_of_this_network = self.get_clustering_coefficient()

        average_shortest_path_of_random_network = math.log(number_of_nodes) / math.log(self.get_average_degree())
        average_shortest_path_of_this_network = self.average_shortest_path_length()

        return (clustering_coefficient_of_this_network - clustering_coefficient_of_random_network) / (average_shortest_path_of_this_network - average_shortest_path_of_random_network)

