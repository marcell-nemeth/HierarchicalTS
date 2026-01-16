"""
Hierarchy Builder Module

This module provides functionality to create hierarchical structures and 
construct the summing matrix for time series reconciliation.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional


class HierarchyBuilder:
    """
    Builds hierarchical structures for time series reconciliation.
    
    The hierarchy consists of:
    - A top node (root)
    - Multiple levels with varying numbers of nodes
    - Bottom-level nodes (base level)
    
    The summing matrix S maps bottom-level forecasts to all levels.
    """
    
    def __init__(self, structure: List[int]):
        """
        Initialize hierarchy builder.
        
        Parameters
        ----------
        structure : List[int]
            Number of nodes at each level from top to bottom.
            Example: [1, 3, 9] means 1 root, 3 middle nodes, 9 bottom nodes.
        """
        self.structure = structure
        self.n_levels = len(structure)
        self.n_nodes = sum(structure)
        self.n_bottom = structure[-1]
        
        # Build hierarchy graph
        self.graph = self._build_graph()
        self.node_info = self._extract_node_info()
        
        # Build summing matrix
        self.S = self._build_summing_matrix()
        
    def _build_graph(self) -> nx.DiGraph:
        """
        Build directed graph representing hierarchy.
        
        Returns
        -------
        nx.DiGraph
            Directed graph with nodes labeled by level and position.
        """
        G = nx.DiGraph()
        
        # Create nodes
        node_id = 0
        node_map = {}  # (level, position) -> node_id
        
        for level, n_nodes in enumerate(self.structure):
            for pos in range(n_nodes):
                node_name = f"L{level}_N{pos}"
                G.add_node(node_id, name=node_name, level=level, position=pos)
                node_map[(level, pos)] = node_id
                node_id += 1
        
        # Create edges (parent-child relationships)
        for level in range(self.n_levels - 1):
            n_parents = self.structure[level]
            n_children = self.structure[level + 1]
            
            # Distribute children among parents as evenly as possible
            children_per_parent = n_children // n_parents
            extra_children = n_children % n_parents
            
            child_idx = 0
            for parent_pos in range(n_parents):
                parent_id = node_map[(level, parent_pos)]
                
                # Number of children for this parent
                n_children_this_parent = children_per_parent
                if parent_pos < extra_children:
                    n_children_this_parent += 1
                
                # Connect to children
                for _ in range(n_children_this_parent):
                    child_id = node_map[(level + 1, child_idx)]
                    G.add_edge(parent_id, child_id)
                    child_idx += 1
        
        return G
    
    def _extract_node_info(self) -> Dict:
        """
        Extract node information from graph.
        
        Returns
        -------
        Dict
            Dictionary mapping node_id to node attributes.
        """
        info = {}
        for node_id in self.graph.nodes():
            attrs = self.graph.nodes[node_id]
            children = list(self.graph.successors(node_id))
            parents = list(self.graph.predecessors(node_id))
            
            info[node_id] = {
                'name': attrs['name'],
                'level': attrs['level'],
                'position': attrs['position'],
                'children': children,
                'parents': parents,
                'is_bottom': attrs['level'] == self.n_levels - 1
            }
        
        return info
    
    def _build_summing_matrix(self) -> np.ndarray:
        """
        Build summing matrix S.
        
        The summing matrix maps bottom-level forecasts to all levels:
        y = S * y_bottom
        
        where:
        - y is the vector of all forecasts (all levels)
        - y_bottom is the vector of bottom-level forecasts
        
        Returns
        -------
        np.ndarray
            Summing matrix of shape (n_nodes, n_bottom)
        """
        S = np.zeros((self.n_nodes, self.n_bottom))
        
        # Get bottom-level node IDs
        bottom_nodes = [
            node_id for node_id, info in self.node_info.items()
            if info['is_bottom']
        ]
        
        # For each node, determine which bottom nodes contribute to it
        for node_id in range(self.n_nodes):
            if self.node_info[node_id]['is_bottom']:
                # Bottom node maps to itself
                bottom_idx = bottom_nodes.index(node_id)
                S[node_id, bottom_idx] = 1.0
            else:
                # Aggregate node: find all descendant bottom nodes
                descendants = self._get_bottom_descendants(node_id)
                for desc_id in descendants:
                    bottom_idx = bottom_nodes.index(desc_id)
                    S[node_id, bottom_idx] = 1.0
        
        return S
    
    def _get_bottom_descendants(self, node_id: int) -> List[int]:
        """
        Get all bottom-level descendants of a node.
        
        Parameters
        ----------
        node_id : int
            Node ID to find descendants for.
            
        Returns
        -------
        List[int]
            List of bottom-level descendant node IDs.
        """
        descendants = []
        
        def dfs(current_node):
            if self.node_info[current_node]['is_bottom']:
                descendants.append(current_node)
            else:
                for child in self.node_info[current_node]['children']:
                    dfs(child)
        
        dfs(node_id)
        return descendants
    
    def get_node_names(self) -> List[str]:
        """Get ordered list of node names."""
        return [self.node_info[i]['name'] for i in range(self.n_nodes)]
    
    def get_bottom_node_names(self) -> List[str]:
        """Get ordered list of bottom-level node names."""
        return [
            self.node_info[i]['name'] for i in range(self.n_nodes)
            if self.node_info[i]['is_bottom']
        ]
    
    def visualize(self) -> Tuple[nx.DiGraph, Dict]:
        """
        Get graph and layout for visualization.
        
        Returns
        -------
        Tuple[nx.DiGraph, Dict]
            Graph and position dictionary for plotting.
        """
        # Create position layout (hierarchical)
        pos = {}
        for node_id, info in self.node_info.items():
            level = info['level']
            position = info['position']
            n_nodes_level = self.structure[level]
            
            # Spread nodes horizontally within level
            x = position - (n_nodes_level - 1) / 2
            y = -level  # Negative so top is at top
            
            pos[node_id] = (x, y)
        
        return self.graph, pos
    
    def get_level_indices(self, level: int) -> List[int]:
        """
        Get node indices for a specific level.
        
        Parameters
        ----------
        level : int
            Level number (0 = top)
            
        Returns
        -------
        List[int]
            List of node IDs at this level.
        """
        return [
            node_id for node_id, info in self.node_info.items()
            if info['level'] == level
        ]
