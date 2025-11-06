from qdrant_client import QdrantClient
from openai import OpenAI
import numpy as np
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from rapidfuzz import fuzz

class HybridSearch:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://qegsy7vzf32snx-8000.proxy.runpod.net/v1",
            api_key="EMPTY"
        )

        # Load Qdrant client
        print("Loading Qdrant client...")
        self.qdrant_client = QdrantClient(
            url="https://qdrant.ml.weaverbase.com:443",
            api_key="db090fa8926fdd3260fb5cf364b9e0b517ea5ab8fdd45f778bb1473603fdfde9",
            timeout=60,
            prefer_grpc=False
        )


    def l2_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize an array of vectors to unit length."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        norms[norms == 0] = 1.0 
        # avoid division by zero     
        return vectors / norms


    def format_for_gritlm(
        self, 
        texts: List[str],
        instruction: Optional[str] = None
    ) -> List[str]:
        """Format texts according to GritLM requirements."""
        if instruction:
            return [
                f"<|user|>\n{instruction}\n<|embed|>\n{text}"
                for text in texts
            ]
        return [f"<|embed|>\n{text}" for text in texts]


    def get_embeddings(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        response = self.client.embeddings.create(
            model="parasail-ai/GritLM-7B-vllm",
            input=texts,
            encoding_format="float"
        )

        tmp_embeddings = [item.embedding for item in response.data]
        # convert tmp_embeddings to numpy array (np.ndarray)
        embeddings_array = np.array(tmp_embeddings)

        norm_embeddings = self.l2_normalize(embeddings_array)

        return norm_embeddings


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True
    )
    def generate_embeddings(
        self,
        texts: List[str],
        instruction: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings with automatic retry.
        
        Args:
            texts: Texts to embed (pre-formatted or raw)
            instruction: Optional instruction for queries
        
        Returns:
            List of 4096-dimensional embeddings
        """
        # Preprocess
        texts = [t.strip() for t in texts]
        
        # Format for GritLM
        formatted = self.format_for_gritlm(texts, instruction)
        
        # logger.info(f"Generating embeddings for {len(texts)} texts")
        start = time.time()
        
        try:
            response = self.client.embeddings.create(
                input=formatted,
                model="parasail-ai/GritLM-7B-vllm"
            )
            
            elapsed = time.time() - start
            # throughput = len(texts) / elapsed
            print(f"elapsed: {elapsed:.2f}s")
            # logger.info(
                # f"Generated {len(texts)} embeddings in {elapsed:.2f}s "
                # f"({throughput:.1f} texts/sec)"
            # )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            # logger.error(f"Embedding generation failed: {e}")
            raise


    def search(
        self,
        query: str,
        instruction: str = "Represent this query for searching relevant passages",
        limit: int = 10
    ):
        """Search for similar vectors."""
        # Generate query embedding
        query_embedding = self.generate_embeddings(
            [query],
            instruction=instruction
        )[0]
        
        # Search Qdrant
        results = self.qdrant_client.search(
            collection_name="ieat_production_embeddings",   # self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=None,
            with_payload=True
        )
        
        return results
    

    def advanced_search(self, query: str, top_k: int = 4) -> List[Dict]:
        threshold = 0.7
        SEMANTIC_WEIGHT = 0.4
        FUZZY_WEIGHT = 0.6
        search_limit = top_k * 5  # Get more candidates for hybrid scoring

        # query_embedding = self.get_embeddings(query)[0]
        
        search_results = self.search(query=query, limit=search_limit)
        # self.qdrant_client.search(
        #     collection_name="similarity_search_poc_cosine",
        #     query_vector=query_embedding.tolist(),
        #     limit=search_limit,
        #     score_threshold=None,  # No semantic threshold
        #     with_vectors=False
        # )

        if not search_results:
            print("No Qdrant results found")
            return []

        
        hybrid_results = []
        for hit in search_results:
            result_item_name = hit.payload['text']
            semantic_score = hit.score
            
            # Calculate fuzzy similarity with query
            fuzzy_score = fuzz.ratio(query.lower(), result_item_name.lower()) / 100
            
            # Hybrid score (weighted combination)
            hybrid_score = (semantic_score * SEMANTIC_WEIGHT) + (fuzzy_score * FUZZY_WEIGHT)
            
            hybrid_results.append({
                'hit': hit,
                'semantic_score': semantic_score,
                'fuzzy_score': fuzzy_score,
                'hybrid_score': hybrid_score
            })
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Filter by hybrid threshold and limit to top_k
        filtered_results = [r for r in hybrid_results if r['hybrid_score'] >= threshold]
        filtered_results = filtered_results[:top_k]
        
        if not filtered_results:
            print(f"\nNo results found with hybrid score >= {threshold}")
            # print(f"(Best hybrid score found: {hybrid_results[0]['hybrid_score']:.4f})" if hybrid_results else "")
            return []

        # Output Results
        output_results = []
        for idx, item in enumerate(filtered_results, 1):
            hit = item['hit']
            # metadata = hit.payload.get("metadata", {})
            # approval_status = metadata.get("approval_status", "no record")
            
            result = {
                "name": hit.payload["text"],
                "semantic_score": round(item['semantic_score'], 4),
                "fuzzy_score": round(item['fuzzy_score'], 4),
                "hybrid_score": round(item['hybrid_score'], 4)
            }
            output_results.append(result)

        return output_results
            
