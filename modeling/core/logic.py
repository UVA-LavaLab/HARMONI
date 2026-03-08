from misc.type import *
from graphviz import Digraph
import networkx as nx
#CLEANUP: model should be a class which can be imported here

# Define the BaseLogicNode class
class BaseLogicNode:
    def __init__(self, name):
        self.name = name
        self.children = []  # List of child nodes
        self.parents = []   # List of parent nodes
        self.connections = []  # List of connections to other nodes (including siblings)
        self.mapped_task = None  # The task mapped to this node

    def add_child(self, child):
        """Add a child node and establish parent-child relationship"""
        child.parents.append(self)
        self.children.append(child)

    def add_connection(self, target_node, weight=1.0, link_latency=0.0, overhead_latency=0.0):
        """Add a connection to another node with weight and latency information"""
        #if target_node not in self.connections:
        if (target_node, weight, link_latency, overhead_latency) not in self.connections:
            self.connections.append((target_node, weight, link_latency, overhead_latency))

    def remove_connection(self, target_node):
        """Remove a connection to another node"""
        self.connections = [(node, weight) for node, weight in self.connections if node != target_node]

    def get_connections(self):
        """Get all nodes this node is connected to"""
        return [node.name for node, _, _, _, _ in self.connections]

    def get_connection_weight(self, target_node):
        """Get the weight of connection to a target node"""
        for node, weight, _, _, _ in self.connections:
            if node == target_node:
                return weight
        return 0.0

    def map_task(self, task):
        """Map a task to this node"""
        self.mapped_task = task

    def get_mapped_task(self):
        """Get the task mapped to this node"""
        return self.mapped_task

    def to_digraph(self, dot=None):
        """Convert the node and its connections to a Digraph"""
        if dot is None:
            dot = Digraph(comment='Logic Node Network')
            dot.attr(rankdir='TB')

        # Add this node
        node_label = f"{self.name}"
        if self.mapped_task:
            node_label += f"\nTask: {self.mapped_task}"
        dot.node(self.name, node_label)

        # Add connections
        for target_node, weight, _, _, _ in self.connections:
            dot.edge(self.name, target_node.name, label=f"{weight:.2f}")

        # Recursively add children
        for child in self.children:
            child.to_digraph(dot)

        return dot

    def to_networkx(self, G=None):
        """Convert the node and its connections to a NetworkX graph"""
        if G is None:
            G = nx.DiGraph()
        if isinstance(self, LogicUnit):
            G.add_node(self.name)
            for target_node, weight, link_latency, overhead_latency in self.connections:
                if isinstance(target_node, LogicUnit):
                    G.add_edge(self.name, target_node.name, weight=weight, link_latency=link_latency, overhead_latency=overhead_latency)
        for child in self.children:
            child.to_networkx(G)
        return G

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, children={len(self.children)}, connections={len(self.connections)})"

class Root(BaseLogicNode):
    pass

class Channel(BaseLogicNode):
    pass


class Rank(BaseLogicNode):
    pass


class Chip(BaseLogicNode):
    pass


class Bankgroup(BaseLogicNode):
    pass


class Bank(BaseLogicNode):
    pass


# Define the LogicUnit class
class LogicUnit(BaseLogicNode):
    def __init__(self, id, num_inbuf, num_outbuf, inbuf_size, outbuf_size, supported_ops, parents):
        self.id = id  # Dictionary with keys: Channel, Rank, Chip, Bankgroup, Bank
        name = f"{'_'.join(f'{k}_{v}' for k, v in id.items())}"
        super().__init__(name=name)
        self.num_inbuf = num_inbuf
        self.num_outbuf = num_outbuf
        self.inbuf_size = inbuf_size  # bytes
        self.outbuf_size = outbuf_size
        self.supported_ops = supported_ops  # List of supported operations
        #self.parent = None  # Reference to parent node (e.g., Bank)
        self.parents = parents
        self.instruction_queue = []  # Queue to store mapped tasks

    def add_task(self, task):
        """Add a task to the instruction queue"""
        self.instruction_queue.append(task)

    def get_next_task(self):
        """Get the next task from the instruction queue"""
        if self.instruction_queue:
            return self.instruction_queue.pop(0)
        return None

    def __repr__(self):
        return (f"LogicUnit(name={self.name}, "
                )
