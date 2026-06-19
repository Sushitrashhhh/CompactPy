def chunk_text_by_words(text: str, chunk_size: int = 50, overlap: int=10) -> list[str]:
    """
    Chunk text by words.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int, optional): The number of words in each chunk. Defaults to 50.
        overlap (int, optional): The number of overlapping words between chunks. Defaults to 10.

    Returns:
        list[str]: A list of text chunks.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i=0
    
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunks.append(" ".join(chunk_words))

        #advance the pointer by chunk_size - overlap to create the next chunk
        i += (chunk_size - overlap)

        #if the next chunk would go beyond the end of the list, break the loop
        if chunk_size <= 0:
            break
    return chunks