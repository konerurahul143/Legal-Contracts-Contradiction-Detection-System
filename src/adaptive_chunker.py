import re
import spacy
from transformers import AutoTokenizer


class AdaptiveChunker:
    """
    Adaptive hierarchical chunker for Legal RAG.

    Strategy

    Clause
        ↓
    Paragraph
        ↓
    Sentence
        ↓
    Token Window

    Goal:
    Preserve legal meaning as much as possible before falling
    back to token-based splitting.
    """

    def __init__(
        self,
        model_name="BAAI/bge-base-en-v1.5",
        max_tokens=350
    ):

        self.max_tokens = max_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Lightweight sentence splitter
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    ####################################################################
    # Token count
    ####################################################################

    def token_count(self, text):

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )

    ####################################################################
    # Paragraph Splitter
    ####################################################################

    def split_paragraphs(self, text):

        paragraphs = re.split(r"\n\s*\n", text)

        return [
            p.strip()
            for p in paragraphs
            if p.strip()
        ]

    ####################################################################
    # Sentence Splitter
    ####################################################################

    def split_sentences(self, text):

        doc = self.nlp(text)

        return [
            sent.text.strip()
            for sent in doc.sents
            if sent.text.strip()
        ]

    ####################################################################
    # Token Window Splitter
    ####################################################################

    def split_by_tokens(self, text):

        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

        chunks = []

        for i in range(0, len(token_ids), self.max_tokens):

            chunk_ids = token_ids[i:i + self.max_tokens]

            chunk = self.tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True
            )

            chunks.append(chunk)

        return chunks

    ####################################################################
    # Adaptive Chunking of One Clause
    ####################################################################

    def chunk_clause(self, clause):

        text = clause["text"].strip()

        # Entire clause fits
        if self.token_count(text) <= self.max_tokens:

            chunk = clause.copy()
            chunk["chunk_id"] = 1

            return [chunk]

        chunks = []

        current_chunk = ""

        chunk_id = 1

        paragraphs = self.split_paragraphs(text)

        for paragraph in paragraphs:

            # Paragraph fits inside current chunk
            if self.token_count(current_chunk + "\n" + paragraph) <= self.max_tokens:

                current_chunk += "\n" + paragraph

            else:

                if current_chunk.strip():

                    chunk = clause.copy()

                    chunk["chunk_id"] = chunk_id
                    chunk["text"] = current_chunk.strip()

                    chunks.append(chunk)

                    chunk_id += 1

                # Huge paragraph
                if self.token_count(paragraph) > self.max_tokens:

                    sentences = self.split_sentences(paragraph)

                    current_chunk = ""

                    for sentence in sentences:

                        if self.token_count(current_chunk + " " + sentence) <= self.max_tokens:

                            current_chunk += " " + sentence

                        else:

                            if current_chunk.strip():

                                chunk = clause.copy()

                                chunk["chunk_id"] = chunk_id
                                chunk["text"] = current_chunk.strip()

                                chunks.append(chunk)

                                chunk_id += 1

                            # Sentence itself huge
                            if self.token_count(sentence) > self.max_tokens:

                                token_chunks = self.split_by_tokens(sentence)

                                for tc in token_chunks:

                                    chunk = clause.copy()

                                    chunk["chunk_id"] = chunk_id
                                    chunk["text"] = tc.strip()

                                    chunks.append(chunk)

                                    chunk_id += 1

                                current_chunk = ""

                            else:

                                current_chunk = sentence

                else:

                    current_chunk = paragraph

        if current_chunk.strip():

            chunk = clause.copy()

            chunk["chunk_id"] = chunk_id
            chunk["text"] = current_chunk.strip()

            chunks.append(chunk)

        return chunks

    ####################################################################
    # Whole Document
    ####################################################################

    def chunk_document(self, clauses):

        document_chunks = []

        for clause in clauses:

            document_chunks.extend(
                self.chunk_clause(clause)
            )

        return document_chunks