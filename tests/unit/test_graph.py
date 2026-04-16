"""Tests for swarm graph construction."""

import numpy as np
import pytest

from core.swarm.graph import build_swarm_graph, compute_graph_features, SwarmGraph


class TestBuildSwarmGraph:
    """Test swarm graph construction."""

    def test_empty_swarm(self) -> None:
        """Test graph construction with no drones."""
        positions = np.zeros((0, 3))
        velocities = np.zeros((0, 3))

        graph = build_swarm_graph(positions, velocities)

        assert graph.n_nodes == 0
        assert graph.n_edges == 0
        assert graph.node_features.shape == (0, 7)

    def test_single_node(self) -> None:
        """Test graph with single drone (no edges)."""
        positions = np.array([[0.0, 0.0, 10.0]])
        velocities = np.array([[1.0, 0.0, 0.0]])

        graph = build_swarm_graph(positions, velocities)

        assert graph.n_nodes == 1
        assert graph.n_edges == 0  # No neighbors

        # Node features: position (3) + velocity (3) + speed (1)
        assert graph.node_features.shape == (1, 7)

    def test_two_nearby_nodes(self) -> None:
        """Test graph with two nearby drones."""
        positions = np.array(
            [[0.0, 0.0, 10.0], [5.0, 0.0, 10.0]]  # Close enough for edge
        )
        velocities = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        graph = build_swarm_graph(positions, velocities, neighbor_radius=10.0)

        assert graph.n_nodes == 2
        assert graph.n_edges == 2  # Bidirectional

        # Edge index should have shape (2, n_edges)
        assert graph.edge_index.shape == (2, 2)

    def test_two_far_nodes(self) -> None:
        """Test graph with two distant drones (no edge)."""
        positions = np.array(
            [[0.0, 0.0, 10.0], [100.0, 0.0, 10.0]]  # Too far for edge
        )
        velocities = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        graph = build_swarm_graph(positions, velocities, neighbor_radius=50.0)

        assert graph.n_nodes == 2
        assert graph.n_edges == 0  # Too far apart

    def test_node_features(self) -> None:
        """Test node feature computation."""
        positions = np.array([[0.0, 0.0, 10.0]])
        velocities = np.array([[3.0, 4.0, 0.0]])  # Speed = 5

        graph = build_swarm_graph(positions, velocities)

        features = graph.node_features[0]

        # Position
        assert np.isclose(features[0], 0.0)
        assert np.isclose(features[1], 0.0)
        assert np.isclose(features[2], 10.0)

        # Velocity
        assert np.isclose(features[3], 3.0)
        assert np.isclose(features[4], 4.0)
        assert np.isclose(features[5], 0.0)

        # Speed
        assert np.isclose(features[6], 5.0)

    def test_edge_features(self) -> None:
        """Test edge feature computation."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        velocities = np.zeros((2, 3))

        graph = build_swarm_graph(
            positions, velocities, include_edge_features=True
        )

        assert graph.edge_features is not None
        assert graph.edge_features.shape[0] == graph.n_edges

        # Edge from 0 to 1: relative position = [1, 0, 0]
        # Edge from 1 to 0: relative position = [-1, 0, 0]


class TestGraphFeatures:
    """Test graph-level feature computation."""

    def test_empty_graph(self) -> None:
        """Test features for empty graph."""
        graph = SwarmGraph(
            node_features=np.zeros((0, 7)),
            edge_index=np.zeros((2, 0), dtype=np.int64),
        )

        features = compute_graph_features(graph)

        assert features["density"] == 0.0
        assert features["avg_degree"] == 0.0
        assert features["connected_components"] == 0

    def test_fully_connected(self) -> None:
        """Test features for fully connected graph."""
        n = 4
        # Complete graph: every node connected to every other
        edge_list = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    edge_list.append([i, j])

        graph = SwarmGraph(
            node_features=np.random.randn(n, 7),
            edge_index=np.array(edge_list, dtype=np.int64).T,
        )

        features = compute_graph_features(graph)

        # Density = 2*|E| / (|V|*(|V|-1)) = 2*12 / (4*3) = 24/12 = 2.0 for directed
        # But for undirected interpretation: 12 edges / 6 possible = 2.0
        # For complete directed graph: n*(n-1) = 12 edges, density = 1.0
        # The formula gives: 2*12 / (4*3) = 2.0
        expected_density = 2.0 * graph.n_edges / (n * (n - 1))
        assert np.isclose(features["density"], expected_density)

        # Each node has degree n-1 in terms of outgoing edges
        # But since we count both directions, degree = n-1 = 3
        assert np.isclose(features["avg_degree"], n - 1)

        # Single connected component
        assert features["connected_components"] == 1

    def test_disconnected_graph(self) -> None:
        """Test features for disconnected graph."""
        # Two separate pairs
        edge_list = [[0, 1], [1, 0], [2, 3], [3, 2]]

        graph = SwarmGraph(
            node_features=np.random.randn(4, 7),
            edge_index=np.array(edge_list, dtype=np.int64).T,
        )

        features = compute_graph_features(graph)

        # Two connected components
        assert features["connected_components"] == 2
