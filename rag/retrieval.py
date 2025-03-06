import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import logging
import pickle
from typing import List


# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('rag.retrieval')



class RagRetriever:

    def __init__(self, media_folder: str):
        self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.index = None
        self.chunks = []
        self.media_folder = Path(media_folder)
        self.index_path = Path(__file__).parent / "faiss_index.bin"
        self.chunks_path = Path(__file__).parent / "chunks.pkl"
        self.initialize_index()



    def initialize_index(self):
        # Check if both files exist
        try:
            if self.index_path.exists() and self.chunks_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                with open(self.chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                logger.debug(f"Loaded {len(self.chunks)} chunks from pickle file")
                return
        except Exception as e:
            logger.error(f"Error loading index and chunks: {e}")

        # If files don't exist, create new index and chunks
        for file_path in self.media_folder.glob('*'):
            if file_path.suffix in ['.txt']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    new_chunks = [chunk.strip() for chunk in content.split('###') if chunk.strip()]
                    self.chunks.extend(new_chunks)
                    logger.debug(f"Loaded {len(new_chunks)} chunks from {file_path}")

        logger.debug(f"Total number of chunks: {len(self.chunks)}")
        
        # Create and save embeddings
        embeddings = self.model.encode(self.chunks)
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        logger.debug(f"Number of vectors in Faiss index: {self.index.ntotal}")

        # Verify normalization
        norms = np.linalg.norm(embeddings, axis=1)
        logger.debug(f"Embedding norms min/max: {norms.min():.3f}/{norms.max():.3f}")

        # Save both index and chunks
        faiss.write_index(self.index, str(self.index_path))
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        logger.debug(f"Saved {len(self.chunks)} chunks to pickle file")





    def get_top_chunks(self, query: str, k: int) -> str:
        
        # First check for direct keyword matches
        direct_matches = self.keyword_lookup(query)
        if direct_matches:
            logger.debug("Using direct keyword match results")
            return "\n\n".join(direct_matches)
    
        # If no direct matches, proceed with vector search
        query_vector = self.model.encode([query])
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(np.array(query_vector).astype('float32'), k)


        threshold = 0
        selected_chunks = []

        for score, idx in zip(scores[0], indices[0]):
            # Log both score and chunk content for debugging
            logger.debug(f"Score: {score:.3f}")
            chunk = self.chunks[idx]
            if score > threshold:
                logger.debug(f"\nSelected chunk (score {score:.3f}): {chunk[:100]}...")
                selected_chunks.append(chunk)
            else:
                logger.debug(f"\nRejected chunk (score {score:.3f}): {chunk[:100]}...")

        if not selected_chunks:
            logger.warning(f"\n\nNo chunks passed threshold {threshold}")
            # Fallback to top result if nothing passes threshold
            return ""

        return "\n\n".join(selected_chunks)
    



    def keyword_lookup(self, query: str) -> List[str]:
        """Find chunks that contain specific keywords mentioned in the query"""
        # Define keywords and their associated chunk indices
        # Format: {keyword: [list of chunk indices that are relevant]}
        keywords = {
            # Education
            'polytechnique': [1, 2, 3, 4, 5, 6],
            'princeton': [0, 1, 2, 3],
            'education': [0, 1, 2, 3, 4, 5, 6],
            'university': [0, 1, 2, 3, 4, 5, 6],
            'courses': [1, 5],
            'academic': [20, 22, 23, 24, 25],
            'degree': [0, 4],
            
            # Work Experience
            'edf': [7, 8, 9, 10],
            'singapore': [7, 8, 9, 10, 20],
            'bain': [11],
            'company': [11],
            'internship': [7, 8, 9, 10, 11],
            'research': [12, 13, 14, 21, 22],
            
            # Skills
            'java': [26],
            'c++': [1, 5, 26],
            'ai': [12, 13, 14, 15, 16],
            'machine learning': [12, 26],
            'deep learning': [5, 9, 22],
            
            # Projects & Leadership
            'project': [12, 13, 14, 21, 22],
            'hackathon': [22],
            'military': [24],
            'leadership': [24],
            'community': [25],
            'volunteering': [25],
            
            # Personal
            'interest': [12, 13, 14, 28],
            'hobby': [28],
            'sport': [28],
            'skydiving': [28],
            'swimming': [28],
            'debate': [23, 28]
        }
        
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Check if any keywords are in the query
        matched_indices = set()
        for keyword, indices in keywords.items():
            if keyword in query_lower:
                logger.debug(f"Keyword match found: {keyword}")
                matched_indices.update(indices)
        
        # Return the chunks corresponding to the matched indices
        matched_chunks = [self.chunks[idx] for idx in matched_indices if idx < len(self.chunks)]
        
        logger.debug(f"Returning {len(matched_chunks)} chunks based on keyword match")
        return matched_chunks