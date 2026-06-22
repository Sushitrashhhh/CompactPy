import networkx as nx

class GraphMemorySystem:
    """
    Version 5+: Graph Memory System.
    Stores knowledge as explicit structural graphs instead of raw, flat text,
    enhanced with custom weights for Phase 3 contextual traversal algorithms.
    """
    def __init__(self):
        # Initialize an empty directed graph using NetworkX
        self.graph = nx.DiGraph()

    def add_relation(self, source: str, relation: str, target: str, weight: float = 1.0):
        """
        Extracts and inserts a directional concept dependency triplet into the graph map,
        including an optional edge weight for advanced ranking.
        """
        source = source.strip()
        target = target.strip()
        relation = relation.strip().lower()

        # Add nodes if they don't exist, then bridge them with a named edge attribute
        self.graph.add_node(source)
        self.graph.add_node(target)
        self.graph.add_edge(source, target, relation=relation, weight=weight)

    def get_relationships_as_text(self) -> list[str]:
        """
        Flattens the graph edges back into hyper-dense structural string representations.
        """
        statements = []
        for u, v, data in self.graph.edges(data=True):
            relation_label = data.get("relation", "connected_to")
            statements.append(f"{u} → {relation_label} → {v}")
        return statements

    def find_connected_concepts(self, entity: str) -> list[str]:
        """
        Retrieves all immediate relational dependencies connected to a specific concept node
        by looking up both upstream predecessors and downstream successors.
        """
        if not self.graph.has_node(entity):
            return []
            
        connections = []
        # Find outgoing relations (Successors)
        for success in self.graph.successors(entity):
            rel = self.graph[entity][success]["relation"]
            connections.append(f"{entity} is {rel} {success}")
            
        # Find incoming relations (Predecessors)
        for pred in self.graph.predecessors(entity):
            rel = self.graph[pred][entity]["relation"]
            connections.append(f"{pred} is {rel} {entity}")
            
        return connections

    def get_graph_summary(self) -> dict[str, int]:
        """Returns fundamental structural dimensions of the memory network."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges()
        }