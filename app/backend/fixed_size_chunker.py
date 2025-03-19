class FixedSizeChunker:
    def __init__(self, chunk_size=100):
        """
        Initializes the chunker with a specified fixed chunk size (number of words).
        
        Parameters:
            chunk_size (int): Number of words per chunk.
        """
        self.chunk_size = chunk_size

    def chunk_text(self, text):
        """
        Splits the provided text into fixed-size chunks.
        
        Parameters:
            text (str): The input text to split.
        
        Returns:
            list: A list of text chunks.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks
