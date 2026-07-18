from collections import defaultdict

from ir.text_processor import tokenize


class InvertedIndex:

    def __init__(self):
        self.index = defaultdict(set)
        self.document_tokens = {}

    def add_document(self, doc_id, text):
        """
        Tokenize a document and add its terms to the inverted index.
        """

        tokens = tokenize(text)

        self.document_tokens[doc_id] = tokens

        for token in set(tokens):
            self.index[token].add(doc_id)

    def build(self, documents):
        """
        Build the full inverted index.
        """

        for doc_id, document in enumerate(documents):
            self.add_document(
                doc_id,
                document
            )

    def search_token(self, token):
        """
        Return document IDs containing a token.
        """

        token = token.lower()

        return self.index.get(
            token,
            set()
        )

    def get_candidates(self, query):
        """
        Retrieve candidate documents containing at least one query token.
        """

        query_tokens = tokenize(query)

        candidates = set()

        for token in query_tokens:
            candidates.update(
                self.index.get(
                    token,
                    set()
                )
            )

        return candidates

    def get_statistics(self):
        """
        Return basic index statistics.
        """

        total_postings = sum(
            len(postings)
            for postings in self.index.values()
        )

        return {
            "unique_terms": len(self.index),
            "indexed_documents": len(
                self.document_tokens
            ),
            "total_postings": total_postings,
        }

    def print_sample(self, limit=20):
        """
        Print sample terms and posting lists.
        """

        print("\nSAMPLE INVERTED INDEX")

        sorted_terms = sorted(
            self.index.keys()
        )

        for term in sorted_terms[:limit]:

            postings = sorted(
                self.index[term]
            )

            preview = postings[:15]

            suffix = (
                " ..."
                if len(postings) > 15
                else ""
            )

            print(
                f"{term:<20} -> "
                f"{preview}{suffix}"
            )