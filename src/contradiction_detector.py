from src.pdf_extractor import PDFExtractor
from src.document_cleaner import DocumentCleaner
from src.clause_segmenter import ClauseSegmenter
from src.adaptive_chunker import AdaptiveChunker
from src.embedder import Embedder
from src.faiss_index import FaissIndex
from src.nli_predictor import NLIPredictor
from src.llm_explainer import LLMExplainer


class ContradictionDetector:

    def __init__(self, model_path):

        print("1. PDFExtractor")
        self.pdf_extractor = PDFExtractor()

        print("2. DocumentCleaner")
        self.cleaner = DocumentCleaner()

        print("3. ClauseSegmenter")
        self.segmenter = ClauseSegmenter()

        print("4. AdaptiveChunker")
        self.chunker = AdaptiveChunker()

        print("5. Embedder")
        self.embedder = Embedder()

        print("6. FAISSIndex")
        self.index = FaissIndex()

        print("7. NLIPredictor")
        self.nli = NLIPredictor(model_path=model_path)

        print("8. LLMExplainer")
        self.llm = LLMExplainer()

        print("Initialization complete")

    def process_documents(self, pdf_paths):

        self.chunks = []

        for pdf_path in pdf_paths:

            document = self.pdf_extractor.extract(pdf_path)

            cleaned_document = self.cleaner.clean_document(document)

            clauses = self.segmenter.segment(cleaned_document)

            chunks = self.chunker.chunk_document(clauses)

            self.chunks.extend(chunks)

        return self.chunks
    
        ####################################################################
    # Build FAISS Vector Index
    ####################################################################

    def build_vector_index(self):
        """
        Generate embeddings for all chunks and build
        the FAISS vector index.
        """

        if not hasattr(self, "chunks") or len(self.chunks) == 0:
            raise ValueError(
                "No chunks available. Run process_documents() first."
            )

        print("\nGenerating embeddings...")

        embeddings = self.embedder.encode_chunks(self.chunks)

        print(f"Generated {len(embeddings)} embeddings.")

        print("\nBuilding FAISS index...")

        self.index.build_index(
            embeddings=embeddings,
            chunks=self.chunks
        )

        print("FAISS index built successfully.")

        return embeddings


    ####################################################################
# Detect Contradictions
####################################################################

    def detect_contradictions(
        self,
        top_k=5,
        contradiction_threshold=0.2
    ):
        """
        Detect contradictions between uploaded documents using

            FAISS Retrieval
                    +
            Fine-tuned DeBERTa NLI
        """

        if self.index.size == 0:
            raise ValueError(
                "FAISS index is empty. Run build_vector_index() first."
            )

        contradictions = []
        seen_pairs = set()

        print("\nSearching for contradictions...")

        for query_chunk in self.chunks:

            ############################################################
            # Create query embedding
            ############################################################

            query_embedding = self.embedder.encode_text(
                query_chunk["text"]
            )

            ############################################################
            # Retrieve similar chunks
            ############################################################

            retrieved_chunks = self.index.search(
                query_embedding=query_embedding,
                top_k=top_k
            )

            ############################################################
            # Compare with retrieved chunks
            ############################################################

            for retrieved_chunk in retrieved_chunks:

               ########################################################
                # Skip comparisons from the same clause
                ########################################################

                if (
                    query_chunk["document"] == retrieved_chunk["document"]
                    and query_chunk["clause_number"] == retrieved_chunk["clause_number"]
                ):
                    continue

                ########################################################
                # Skip identical chunk
                ########################################################

                if (
                    query_chunk["document"] == retrieved_chunk["document"]
                    and query_chunk["page_start"] == retrieved_chunk["page_start"]
                    and query_chunk["chunk_id"] == retrieved_chunk["chunk_id"]
                ):
                    continue

                ########################################################
                # Run NLI
                ########################################################

                prediction = self.nli.predict(
                    premise=query_chunk["text"],
                    hypothesis=retrieved_chunk["text"]
                )

                ########################################################
                # Keep contradictions only
                ########################################################

                if (
                    prediction["label"] == "Contradiction"
                    and prediction["confidence"] >= contradiction_threshold
                ):
                    
                    ########################################################
                    # Remove duplicate pairs
                    ########################################################

                    pair = tuple(sorted([
                        (
                            query_chunk["document"],
                            query_chunk["clause_number"],
                            query_chunk["chunk_id"]
                        ),
                        (
                            retrieved_chunk["document"],
                            retrieved_chunk["clause_number"],
                            retrieved_chunk["chunk_id"]
                        )
                    ]))

                    if pair in seen_pairs:
                        continue

                    seen_pairs.add(pair)

                    contradictions.append({

                        "source_document":
                            query_chunk["document"],

                        "source_clause":
                            query_chunk["clause_number"],

                        "source_page":
                            query_chunk["page_start"],

                        "source_chunk":
                            query_chunk["chunk_id"],

                        "source_text":
                            query_chunk["text"],

                        "target_document":
                            retrieved_chunk["document"],

                        "target_clause":
                            retrieved_chunk["clause_number"],

                        "target_page":
                            retrieved_chunk["page_start"],

                        "target_chunk":
                            retrieved_chunk["chunk_id"],

                        "target_text":
                            retrieved_chunk["text"],

                        "similarity":
                            retrieved_chunk["score"],

                        "label":
                            prediction["label"],

                        "confidence":
                            prediction["confidence"],

                        "explanation": self.llm.explain(
                            source_text=query_chunk["text"],
                            target_text=retrieved_chunk["text"]
                        )
                    })

        self.contradictions = contradictions

        print(f"\nDetected {len(contradictions)} contradictions.")

        return contradictions
  
