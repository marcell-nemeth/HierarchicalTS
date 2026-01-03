import numpy as np
import scipy.sparse as sp
import networkx as nx

class Hierarchy:
    """
    Manages the spatial hierarchy structure (tree graph).
    """
    def __init__(self, structure):
        """
        Initialize the hierarchy.
        
        Args:
            structure: A dictionary representing the tree. 
                       Keys are parent nodes, values are lists of children.
                       Leaf nodes should form the bottom level.
        """
        self.structure = structure
        self.graph = self._build_graph(structure)
        self.nodes = list(self.graph.nodes())
        self.n_nodes = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.bottom_nodes = [n for n in self.nodes if self.graph.out_degree(n) == 0]
        self.m_bottom = len(self.bottom_nodes)

    def get_bottom_indices(self):
        """Returns the indices of bottom-level nodes in the node list."""
        return [self.node_to_idx[n] for n in self.bottom_nodes]
        
    def _build_graph(self, structure):
        """Constructs a NetworkX DiGraph from the structure dict."""
        G = nx.DiGraph()
        # Ensure a root exists if implicit, but here we assume explicit structure
        # Traverse and add edges
        def add_edges(parent, children):
            for child in children:
                G.add_edge(parent, child)
                if child in structure:
                    add_edges(child, structure[child])
        
        # Find roots (nodes in keys but not in values)
        all_children = set()
        for children in structure.values():
            all_children.update(children)
        roots = [n for n in structure.keys() if n not in all_children]
        
        for root in roots:
            add_edges(root, structure[root])
            
        return G

    def get_summing_matrix(self):
        """
        Returns the purely spatial Summing Matrix S_sp (n_s x m_s).
        Rows correspond to all nodes, columns to bottom nodes.
        S_sp[i, j] = 1 if bottom node j aggregates into node i.
        """
        S = np.zeros((self.n_nodes, self.m_bottom))
        
        # Precompute ancestors for all bottom nodes
        # In a tree, the path is unique. S[i, j] = 1 iff i is ancestor of j (or i==j)
        # However, networkx algorithms might be slow for huge graphs, but efficient enough here.
        
        # Optimization: Use reachability
        # Since it's a hierarchy, we can just trace up from each bottom node to root
        
        # Build parent map for fast traversal
        parent_map = {}
        for parent, children in self.structure.items():
            for child in children:
                parent_map[child] = parent

        for j, b_node in enumerate(self.bottom_nodes):
            current = b_node
            while True:
                idx = self.node_to_idx[current]
                S[idx, j] = 1
                if current in parent_map:
                    current = parent_map[current]
                else:
                    break # Root reached (or disconnected component)
        
        return S

    def get_dummy_summing_matrix(self):
        """
        Actually, typical S matrices are ordered such that top are aggregates, bottom are bottom.
        This method is just a wrapper for likely standard usage where we might want the explicit S matrix.
        """
        return self.get_summing_matrix()

    def get_graph_laplacian(self):
        """
        Returns the unnormalized Graph Laplacian L = D - A.
        Ordering of rows/cols corresponds to self.nodes.
        """
        # We need the UNDIRECTED graph for the Laplacian in the context of MRF usually,
        # or at least symmetric adjacency for distance definition. 
        # Geodesic distance is on the "skeleton" of the tree.
        G_undir = self.graph.to_undirected()
        L = nx.laplacian_matrix(G_undir, nodelist=self.nodes)
        return L

    def get_geodesic_distance_matrix(self):
        """
        Computes pairwise geodesic distances on the tree.
        Returns a dense matrix (n_s x n_s). 
        WARNING: O(N^2) storage. For truly massive graphs, we wouldn't form this full matrix,
        but GS-GLS uses the Laplacian precision directly. This is mainly for:
        1. Theoretical verification
        2. Small-scale demo
        3. Estimating parameters if we used the dense kernel approach.
        """
        G_undir = self.graph.to_undirected()
        length = dict(nx.all_pairs_shortest_path_length(G_undir))
        D = np.zeros((self.n_nodes, self.n_nodes))
        for i, u in enumerate(self.nodes):
            for j, v in enumerate(self.nodes):
                D[i, j] = length[u][v]
        return D
