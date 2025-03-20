from nltk.tokenize import sent_tokenize
import nltk

# Download necessary NLTK package
nltk.download('punkt')

def get_text_chunks(file_path, max_length=256):
    """
    Reads a markdown file, tokenizes it into sentences, and returns sentence-based chunks.
    
    Parameters:
        file_path (str): Path to the markdown file.
        max_length (int): Maximum length of each chunk (default: 256).
    
    Returns:
        list: List of sentence-based chunks.
    """
    # Load the markdown file
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # Split text into sentences
    sentences = sent_tokenize(markdown_text)

    chunks = []
    current_chunk = ""
    max_length = 256  # Set your max length

    for sentence in sentences:
        if len(sentence) > max_length:
            # If a single sentence exceeds max_length, append it separately
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.append(sentence.strip())
        elif len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())


    return chunks

# Example usage
file_path = "/Users/janvichitroda/Documents/Janvi/NEU/Big_Data_Intelligence_Analytics/Assignment 4/Part 2/LLM_With_Pinecone/PineconeHandson/InputFiles/inputFile.md"
chunks = get_text_chunks(file_path)

# Print the chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")
