import re
import nltk
from typing import List, Optional

# Download nltk data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize

class DocumentChunker:
    """
    A unified class that provides multiple chunking strategies for document text.
    Specifically designed for markdown content extracted from financial reports.
    """
    
    def __init__(self, fixed_chunk_size: int = 100, max_sentence_length: int = 256):
        """
        Initialize the document chunker.
        
        Args:
            fixed_chunk_size: Number of words for fixed-size chunking
            max_sentence_length: Maximum character length for sentence-based chunks
        """
        self.fixed_chunk_size = fixed_chunk_size
        self.max_sentence_length = max_sentence_length
    
    def markdown_header_chunks(self, text: str) -> List[str]:
        """
        Chunk text based on markdown headers.
        
        Args:
            text: The markdown text to chunk.
            
        Returns:
            List of text chunks with headers as separation points.
        """
        # Regular expression to find markdown headers
        header_pattern = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)
        
        # Find all headers and their positions
        headers = [(match.start(), match.group()) for match in header_pattern.finditer(text)]
        
        # If no headers found, return whole text as one chunk
        if not headers:
            return [text.strip()]
        
        chunks = []
        
        # First chunk includes everything before the first header
        if headers[0][0] > 0:
            chunks.append(text[:headers[0][0]].strip())
        
        # Process chunks between headers
        for i in range(len(headers)):
            start_pos = headers[i][0]
            # If this is the last header, end position is the end of text
            if i == len(headers) - 1:
                end_pos = len(text)
            else:
                end_pos = headers[i+1][0]
            
            # Add the chunk, including the header
            chunk = text[start_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)
        
        return [chunk for chunk in chunks if chunk]  # Remove any empty chunks
    
    def sentence_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks based on sentences.
        
        Args:
            text: The text to split into sentence-based chunks.
            
        Returns:
            List of sentence-based chunks.
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If a single sentence exceeds max_length, append it separately
            if len(sentence) > self.max_sentence_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(sentence.strip())
            # If adding the sentence would keep the chunk under max_length
            elif len(current_chunk) + len(sentence) + 1 <= self.max_sentence_length:
                current_chunk += " " + sentence if current_chunk else sentence
            # Otherwise, start a new chunk
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def fixed_size_chunks(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks based on word count.
        
        Args:
            text: The text to split into fixed-size chunks.
            
        Returns:
            List of fixed-size chunks.
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.fixed_chunk_size):
            chunk = " ".join(words[i:i + self.fixed_chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def chunk_document(self, text: str, strategy: str = "markdown") -> List[str]:
        """
        Chunk a document using the specified strategy.
        
        Args:
            text: The document text to chunk.
            strategy: The chunking strategy to use ("markdown", "sentence", or "fixed").
            
        Returns:
            List of text chunks.
        """
        if strategy == "markdown":
            return self.markdown_header_chunks(text)
        elif strategy == "sentence":
            return self.sentence_chunks(text)
        elif strategy == "fixed":
            return self.fixed_size_chunks(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")