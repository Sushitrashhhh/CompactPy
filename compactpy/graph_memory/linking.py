from compactpy.graph_memory.graph import GraphMemorySystem
from compactpy.graph_memory.parser import RelationshipExtractor

class GraphMemoryLinker:
    """
    Coordinates text extraction and graph injection to build out 
    the unified Graph Memory System automatically from raw stream context.
    """
    def __init__(self, memory_system: GraphMemorySystem):
        self.gms = memory_system
        self.extractor = RelationshipExtractor()

    def ingest_unstructured_text(self, text: str) -> int:
        """
        Parses an unstructured text block, extracts relational matches, 
        and links them directly inside the network. Returns total linkages made.
        """
        triplets = self.extractor.extract_triplets_from_text(text)
        for source, relation, target in triplets:
            # Set a high base weight for explicit text extractions
            self.gms.add_relation(source, relation, target, weight=1.0)
        return len(triplets)