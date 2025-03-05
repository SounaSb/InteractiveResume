import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import logging
import pickle


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)




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





    def get_top_chunks(self, query: str, k: int = 3) -> str:
        query_vector = self.model.encode([query])
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(np.array(query_vector).astype('float32'), k)


        threshold = 0.25
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
            selected_chunks = [self.chunks[indices[0][0]]]

        return "\n\n".join(selected_chunks)
    
