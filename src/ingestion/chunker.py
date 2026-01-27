from typing import List, Dict, Any, Optional
import uuid
from ..core.document import DocumentNode
from .splitter import SimpleTextSplitter


class HierarchicalChunker:

    def __init__(self, parent_size=2000, child_size=400, overlap=50):
        self.parent_splitter = SimpleTextSplitter(parent_size, overlap)
        self.child_splitter = SimpleTextSplitter(child_size, overlap)

    def build_chunks(self, node: DocumentNode):
        """
        Returns:
          parents: List[Dict]
          children: List[Dict]
        """
        parents = []
        children = []

        # chỉ materialize node đủ lớn (section-level)
        if node.text.strip():
            parent_chunks = self.parent_splitter.split_text(node.text)

            for p_text in parent_chunks:
                parent_id = str(uuid.uuid4())
                parents.append({
                    "id": parent_id,
                    "text": p_text,
                    "node_id": node.id,
                    "level": node.level
                })

                child_chunks = self.child_splitter.split_text(p_text)
                for c_text in child_chunks:
                    children.append({
                        "text": c_text,
                        "parent_id": parent_id,
                        "node_id": node.id
                    })

        for child in node.children:
            p, c = self.build_chunks(child)
            parents.extend(p)
            children.extend(c)

        return parents, children
