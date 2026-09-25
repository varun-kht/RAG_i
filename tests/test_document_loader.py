"""
Unit tests for DocumentLoader and RecursiveTextSplitter.
"""

import unittest
from document_loader import Document, RecursiveTextSplitter

class TestDocumentLoader(unittest.TestCase):

    def test_recursive_splitter(self):
        content = "Paragraph 1 is here with some details.\n\nParagraph 2 contains more context.\n\nParagraph 3 ends the test document."
        doc = Document(content=content, metadata={"source": "test.txt"}, doc_id="test.txt")
        
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split_document(doc)
        
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].doc_id, "test.txt")
        self.assertIn("Paragraph", chunks[0].text)

if __name__ == "__main__":
    unittest.main()
