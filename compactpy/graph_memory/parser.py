import re

class RelationshipExtractor:
    """
    Parses unstructured contextual strings into clean, relational 
    knowledge triplets (Source, Relation, Target) for graph injection.
    """
    def __init__(self):
        self.relation_keywords = ["utilizes", "deploys", "leverages", "extends", "implements", "powers", "causes"]
        # Core entities to explicitly normalize for our knowledge network
        self.known_entities = ["mediscan ai", "fastapi", "container clusters", "microservices", "compactpy", "bm25 engine"]

    def _normalize_entity(self, entity_text: str) -> str:
        """Finds known core technical entities inside messy strings to merge graph nodes cleanly."""
        cleaned = entity_text.strip().lower()
        for entity in self.known_entities:
            if entity in cleaned:
                # FIX: Preserve spaces between capitalized words
                return " ".join([word.capitalize() if word.lower() != "ai" else "AI" for word in entity.split()])
        
        # Fallback to cleaning punctuation if no known entity matches
        return re.sub(r'[^\w\s-]', '', entity_text).strip()

    def extract_triplets_from_text(self, text: str) -> list[tuple[str, str, str]]:
        """Scans text for relationship keywords and normalizes entities to build clean chains."""
        triplets = []
        clauses = re.split(r'[.,;\n]', text)
        
        for clause in clauses:
            clause = clause.strip()
            for kw in self.relation_keywords:
                if f" {kw} " in clause.lower():
                    pattern = f"(?i) {kw} "
                    parts = re.split(pattern, clause, maxsplit=1)
                    if len(parts) == 2:
                        source = self._normalize_entity(parts[0])
                        target = self._normalize_entity(parts[1])
                        if source and target:
                            triplets.append((source, kw, target))
        return triplets