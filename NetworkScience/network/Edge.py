
edges_created = 0

class Edge:
    def __init__(self, node_a, node_b):
        global edges_created
        self.node_a = node_a
        self.node_b = node_b
        self.key = edges_created + 1
        edges_created += 1

    def rewire(self, node_a, node_b):
        self.node_a = node_a
        self.node_b = node_b

    def traverse(self, node):
        if node == self.node_a:
            return self.node_b
        return self.node_a

    def __eq__(self, other):
        return self.key == other.key