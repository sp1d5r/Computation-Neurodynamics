from .Edge import Edge

class Node:
    def __init__(self, key):
        self.key = key
        self.edges = []

    def add_edge(self, edge):
        self.edges.append(edge)

    def connect_to_node(self, node):
        edge = Edge(self, node)
        node.add_edge(edge)

    def remove_edge(self, edge):
        for i, e in enumerate(self.edges):
            if e == edge:
                self.edges.pop(i)

    def get_neighbours(self):
        neighbours = []
        for edge in self.edges:
            neighbours.append(edge.traverse(self))
        return neighbours

    def is_connected(self, node_b):
        neighbours = self.get_neighbours()
        return node_b in neighbours

    def get_degree(self):
        return len(self.edges)

    def __eq__(self, other):
        return self.key == other.key