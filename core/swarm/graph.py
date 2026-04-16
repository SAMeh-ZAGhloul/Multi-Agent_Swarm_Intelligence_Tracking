"""Swarm graph representation for GNN processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..constants import NEIGHBORHOOD_RADIUS

Vec3 = npt.NDArray[np.float64]
Mat = npt.NDArray[np.float64]


@dataclass
class SwarmGraph:
    """
    Graph representation of a drone swarm.

    This dataclass holds the graph structure used by GNN models.
    It follows PyTorch Geometric conventions.

    Attributes
    ----------
    node_features : npt.NDArray
        Node feature matrix, shape (n_nodes, n_features).
    edge_index : npt.NDArray
        Edge connectivity, shape (2, n_edges).
    edge_features : npt.NDArray | None
        Optional edge features, shape (n_edges, n_edge_features).
    n_nodes : int
        Number of nodes in the graph.
    n_edges : int
        Number of edges in the graph.
    """

    node_features: npt.NDArray[np.float64]
    edge_index: npt.NDArray[np.int64]
    edge_features: npt.NDArray[np.float64] | None = None
    n_nodes: int = 0
    n_edges: int = 0

    def __post_init__(self) -> None:
        """Validate and set derived attributes."""
        self.n_nodes = self.node_features.shape[0]
        self.n_edges = self.edge_index.shape[1] if self.edge_index.size > 0 else 0


def build_swarm_graph(
    positions: list[Vec3] | npt.NDArray[np.float64],
    velocities: list[Vec3] | npt.NDArray[np.float64],
    neighbor_radius: float = NEIGHBORHOOD_RADIUS,
    include_edge_features: bool = True,
) -> SwarmGraph:
    """
    Build a graph representation from swarm positions and velocities.

    Parameters
    ----------
    positions : list[Vec3] | npt.NDArray
        Drone positions, shape (n_drones, 3).
    velocities : list[Vec3] | npt.NDArray
        Drone velocities, shape (n_drones, 3).
    neighbor_radius : float
        Maximum distance for edge creation.
    include_edge_features : bool
        Whether to compute edge features.

    Returns
    -------
    SwarmGraph
        Graph representation of the swarm.

    Notes
    -----
    Node features (7 per node):
        - position (3): [px, py, pz]
        - velocity (3): [vx, vy, vz]
        - speed (1): ||v||

    Edge features (3 per edge):
        - relative_position (3): p_j - p_i
    """
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)

    n_nodes = positions.shape[0]
    if n_nodes == 0:
        return SwarmGraph(
            node_features=np.zeros((0, 7), dtype=np.float64),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_features=None,
        )

    # Compute node features
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    node_features = np.concatenate([positions, velocities, speeds], axis=1)

    # Build adjacency (edges)
    edge_list = []
    edge_features_list = []

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < neighbor_radius:
                # Add bidirectional edges
                edge_list.append([i, j])
                edge_list.append([j, i])

                if include_edge_features:
                    rel_pos = positions[j] - positions[i]
                    edge_features_list.append(rel_pos)
                    edge_features_list.append(-rel_pos)  # opposite direction

    if len(edge_list) == 0:
        # No edges - return empty graph
        return SwarmGraph(
            node_features=node_features,
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_features=None,
        )

    edge_index = np.array(edge_list, dtype=np.int64).T

    if include_edge_features and len(edge_features_list) > 0:
        edge_features = np.array(edge_features_list, dtype=np.float64)
    else:
        edge_features = None

    return SwarmGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        n_nodes=n_nodes,
        n_edges=len(edge_list),
    )


def compute_graph_features(graph: SwarmGraph) -> dict:
    """
    Compute graph-level features from swarm graph.

    Parameters
    ----------
    graph : SwarmGraph
        Input swarm graph.

    Returns
    -------
    dict
        Graph-level features including density, clustering, etc.
    """
    if graph.n_nodes == 0:
        return {
            "density": 0.0,
            "avg_degree": 0.0,
            "connected_components": 0,
        }

    # Degree of each node
    degree = np.zeros(graph.n_nodes, dtype=np.int64)
    for i in range(graph.n_edges):
        degree[graph.edge_index[0, i]] += 1

    avg_degree = float(np.mean(degree))

    # Graph density: 2 * |E| / (|V| * (|V| - 1))
    max_edges = graph.n_nodes * (graph.n_nodes - 1)
    density = (2 * graph.n_edges / max_edges) if max_edges > 0 else 0.0

    # Connected components (simple BFS)
    visited = np.zeros(graph.n_nodes, dtype=bool)
    n_components = 0

    # Build adjacency list
    adj: list[list[int]] = [[] for _ in range(graph.n_nodes)]
    for i in range(graph.n_edges):
        src = graph.edge_index[0, i]
        dst = graph.edge_index[1, i]
        adj[src].append(dst)

    for start in range(graph.n_nodes):
        if visited[start]:
            continue
        n_components += 1
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

    return {
        "density": density,
        "avg_degree": avg_degree,
        "connected_components": n_components,
    }
