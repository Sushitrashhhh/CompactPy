import tiktoken

def count_tokens(text: str, model_name: str='gpt-4') -> int:
    """
    Count the number of tokens in a given text using the specified model's tokenizer.

    Args:
        text (str): The input text to be tokenized and counted.
        model_name (str, optional): The name of the model whose tokenizer to use. Defaults to 'gpt-4'.

    Returns:
        int: The number of tokens in the input text.
    """
    try: # get the standard encoding for the specified model
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError: # if the model is not found, use the default encoding
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def calculate_savings(original_text: str, compressed_text: str, model_name: str='gpt-4') -> dict:
    """
    Calculate the percentage of tokens saved by compressing the original text.

    Args:
        original_text (str): The original input text before compression.
        compressed_text (str): The compressed version of the input text.
        model_name (str, optional): The name of the model whose tokenizer to use. Defaults to 'gpt-4'.

    Returns:
        float: The percentage of tokens saved by compression.
    """
    orig_tokens = count_tokens(original_text, model_name)
    comp_tokens = count_tokens(compressed_text, model_name)
    tokens_saved = orig_tokens - comp_tokens
    
    reduction_percentage = (tokens_saved / orig_tokens * 100) if orig_tokens > 0 else 0.0
    
    # CRITICAL: Check that this matches exactly and returns a dictionary
    return {
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "tokens_saved": tokens_saved,
        "reduction_percentage": round(reduction_percentage, 2)
    }