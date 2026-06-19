from compactpy.token_utils import calculate_savings

class DeduplicationEngine:
    '''
    v1: deduplication engine that identifies and removes duplicate text segments from a given input.
    '''
    def compress(self, text:list[str]) -> tuple[list[str], dict]:
        """
        Compress the input text by removing duplicate segments.

        Args:
            text (list[str]): A list of text segments to be compressed.

        Returns:
            tuple: A tuple containing the compressed text and a report dictionary.
        """
        seen = set()
        deduplicated = []
        
        for segment in text:
            if segment not in seen:
                seen.add(segment)
                deduplicated.append(segment)
        
        # Generate a report on token savings
        original_text = " ".join(text)
        deduplicated_str = " ".join(deduplicated)
        report = calculate_savings(original_text, deduplicated_str)
        
        return deduplicated, report