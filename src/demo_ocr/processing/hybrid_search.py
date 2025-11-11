from qdrant_client import QdrantClient
from openai import OpenAI
import numpy as np
import uuid
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, 
    HnswConfigDiff, OptimizersConfigDiff
)
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from rapidfuzz import fuzz

class HybridSearch:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://2vnn6llpqzyyo8-8000.proxy.runpod.net/v1",
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
    def upload_batch_to_qdrant(
        self,
        points: List[PointStruct],
        wait: bool = False
    ):
        """Upload batch with retry logic."""
        self.qdrant_client.upsert(
            collection_name="ieat_production_embeddings",
            points=points,
            wait=wait
        )

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


    def process_dataset(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
        instruction: Optional[str] = None,
        embedding_batch_size: int = 32,
        upload_batch_size: int = 32
    ) -> List[str]:
        """
        Process complete dataset: embed and store in Qdrant.
        
        Args:
            texts: All texts to process
            metadata: Optional metadata for each text
            instruction: Embedding instruction (for queries)
            embedding_batch_size: Batch size for embedding generation
            upload_batch_size: Batch size for Qdrant upload
        
        Returns:
            List of point IDs in Qdrant
        """
        if metadata is None:
            metadata = [{}] * len(texts)
        
        total_texts = len(texts)
        all_point_ids = []
        
        # logger.info(f"Processing {total_texts} texts")
        # logger.info(f"Embedding batch size: {embedding_batch_size}")
        # logger.info(f"Upload batch size: {upload_batch_size}")
        
        # Process in embedding batches
        for i in range(0, total_texts, embedding_batch_size):
            batch_texts = texts[i:i+embedding_batch_size]
            batch_meta = metadata[i:i+embedding_batch_size]
            
            batch_num = i // embedding_batch_size + 1
            total_batches = (total_texts - 1) // embedding_batch_size + 1
            
            # logger.info(f"Processing embedding batch {batch_num}/{total_batches}")
            
            # Generate embeddings
            try:
                embeddings = self.generate_embeddings(batch_texts, instruction)
            except Exception as e:
                # logger.error(f"Failed to generate embeddings for batch {batch_num}: {e}")
                continue
            
            # Create points
            batch_point_ids = [str(uuid.uuid4()) for _ in batch_texts]
            points = [
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"text": text, **meta}
                )
                for point_id, text, embedding, meta
                in zip(batch_point_ids, batch_texts, embeddings, batch_meta)
            ]
            
            all_point_ids.extend(batch_point_ids)
            
            # Upload to Qdrant in sub-batches
            for j in range(0, len(points), upload_batch_size):
                upload_batch = points[j:j+upload_batch_size]
                
                try:
                    self.upload_batch_to_qdrant(
                        upload_batch,
                        wait=False  # Async for better throughput
                    )
                except Exception as e:
                    # logger.error(f"Failed to upload batch: {e}")
                    continue
            
            # Brief pause between embedding batches
            if i + embedding_batch_size < total_texts:
                time.sleep(0.1)
        
        # logger.info(f"Completed processing {len(all_point_ids)} texts")
        time.sleep(3)
        self.qdrant_client.update_collection(
            collection_name="ieat_production_embeddings",
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=20000)
        )
        

        return all_point_ids

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
                "doc_id": hit.payload["doc_id"],
                "semantic_score": round(item['semantic_score'], 4),
                "fuzzy_score": round(item['fuzzy_score'], 4),
                "hybrid_score": round(item['hybrid_score'], 4)
            }
            output_results.append(result)

        return output_results
            
