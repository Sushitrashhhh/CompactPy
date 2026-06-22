from compactpy.graph_memory.graph import GraphMemorySystem

class GraphTraversalEngine:
    """
    Executes advanced graph analytics and multi-hop path traversals
    to uncover deep semantic relationship webs within the memory system.
    """
    def __init__(self, memory_system: GraphMemorySystem):
        self.gms = memory_system

    def traverse_subgraph_context(self, start_entity: str, max_depth: int = 2) -> list[str]:
        """
        Performs a Breadth-First Search (BFS) up to a max_depth to extract
        an entire interconnected structural context pool for an entity.
        """
        graph = self.gms.graph
        if not graph.has_node(start_entity):
            return []

        visited = set()
        queue = [(start_entity, 0)]
        extracted_context = []

        while queue:
            current_node, depth = queue.pop(0)
            
            if current_node in visited:
                continue
            visited.add(current_node)

            if depth >= max_depth:
                continue

            neighbors = list(graph.successors(current_node)) + list(graph.predecessors(current_node))
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    if graph.has_edge(current_node, neighbor):
                        rel = graph[current_node][neighbor]["relation"]
                        extracted_context.append(f"{current_node} → {rel} → {neighbor}")
                    elif graph.has_edge(neighbor, current_node):
                        rel = graph[neighbor][current_node]["relation"]
                        extracted_context.append(f"{neighbor} → {rel} → {current_node}")
                    
                    queue.append((neighbor, depth + 1))

        return list(dict.fromkeys(extracted_context))