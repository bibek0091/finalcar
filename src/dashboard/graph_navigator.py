"""
graph_navigator.py — BFMC Track Graph & A* Path Planning
=========================================================
Parses Track.graphml, runs A* pathfinding, and handles
sign-based localization for the BFMC dashboard.
"""

import os
import math
import heapq
import json
import xml.etree.ElementTree as ET


class TrackGraph:
    """Directed graph parsed from Track.graphml with A* pathfinding."""

    def __init__(self, graphml_path=None):
        if graphml_path is None:
            graphml_path = os.path.join(os.path.dirname(__file__), "Track.graphml")
        self.nodes = {}       # {id_str: {"x": float, "y": float}}
        self.edges = []       # [{"src": str, "dst": str, "dotted": bool}]
        self.adj = {}         # adjacency list: {src: [(dst, weight), ...]}
        self._parse(graphml_path)

    # ── Parsing ──────────────────────────────────────────────────────────
    def _parse(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

        for node_el in root.findall(".//g:node", ns):
            nid = node_el.get("id")
            x = y = 0.0
            for data in node_el.findall("g:data", ns):
                key = data.get("key")
                if key == "d0":
                    x = float(data.text)
                elif key == "d1":
                    y = float(data.text)
            self.nodes[nid] = {"x": x, "y": y}

        for edge_el in root.findall(".//g:edge", ns):
            src = edge_el.get("source")
            dst = edge_el.get("target")
            dotted = False
            for data in edge_el.findall("g:data", ns):
                if data.get("key") == "d2":
                    dotted = data.text.strip().lower() == "true"
            self.edges.append({"src": src, "dst": dst, "dotted": dotted})

        # Build adjacency list with Euclidean distance weights
        for nid in self.nodes:
            self.adj[nid] = []
        for e in self.edges:
            s, d = e["src"], e["dst"]
            if s in self.nodes and d in self.nodes:
                w = self._dist(s, d)
                self.adj[s].append((d, w))

    def _dist(self, a, b):
        ax, ay = self.nodes[a]["x"], self.nodes[a]["y"]
        bx, by = self.nodes[b]["x"], self.nodes[b]["y"]
        return math.hypot(bx - ax, by - ay)

    # ── A* Pathfinding ───────────────────────────────────────────────────
    def find_route(self, start_id, end_id):
        """A* shortest path. Returns list of node IDs or [] if no path."""
        start_id, end_id = str(start_id), str(end_id)
        if start_id not in self.nodes or end_id not in self.nodes:
            return []

        open_set = [(0.0, start_id)]  # (f_score, node_id)
        came_from = {}
        g_score = {start_id: 0.0}
        f_score = {start_id: self._dist(start_id, end_id)}
        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == end_id:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            for neighbor, weight in self.adj.get(current, []):
                if neighbor in visited:
                    continue
                tentative_g = g_score[current] + weight
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._dist(neighbor, end_id)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    # ── Sign-Based Localization ──────────────────────────────────────────
    def find_nearest_node(self, x, y):
        """Returns the node ID closest to (x, y)."""
        best_id, best_dist = None, float("inf")
        for nid, pos in self.nodes.items():
            d = math.hypot(pos["x"] - x, pos["y"] - y)
            if d < best_dist:
                best_dist = d
                best_id = nid
        return best_id

    def update_car_position(self, detected_sign_label, sign_placements, route):
        """
        Given a detected YOLO sign label and user-placed signs on the route,
        returns the (node_id, x, y) of the matching sign position.
        
        sign_placements: [{"label": str, "node_id": str}, ...]
        route: [node_id, ...] from A*
        """
        for sp in sign_placements:
            if sp["label"] == detected_sign_label and sp["node_id"] in route:
                nid = sp["node_id"]
                pos = self.nodes.get(nid, {"x": 0, "y": 0})
                return {"node_id": nid, "x": pos["x"], "y": pos["y"]}
        return None

    # ── JSON Export for Frontend ─────────────────────────────────────────
    def to_json(self):
        """Export graph as JSON dict for frontend rendering."""
        return {
            "nodes": {nid: {"x": p["x"], "y": p["y"]} for nid, p in self.nodes.items()},
            "edges": self.edges,
        }

    def route_coords(self, route):
        """Convert route (list of node IDs) to list of {x, y} coords."""
        return [
            {"id": nid, "x": self.nodes[nid]["x"], "y": self.nodes[nid]["y"]}
            for nid in route if nid in self.nodes
        ]


# ── Module-level singleton ───────────────────────────────────────────────
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = TrackGraph()
    return _graph


if __name__ == "__main__":
    g = TrackGraph()
    print(f"Loaded: {len(g.nodes)} nodes, {len(g.edges)} edges")
    # Test A* between first and last node
    nids = sorted(g.nodes.keys(), key=int)
    if len(nids) >= 2:
        route = g.find_route(nids[0], nids[-1])
        print(f"A* route from {nids[0]} to {nids[-1]}: {len(route)} steps")
        if route:
            print(f"  First 10: {route[:10]}")
