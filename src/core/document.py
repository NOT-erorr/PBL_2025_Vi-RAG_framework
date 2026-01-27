from dataclasses import dataclass, field
from typing import List, Optional, Dict
import uuid

@dataclass
class DocumentNode:
    """
    modelize a Document to Tree with structure:
    Document
    ├── Section 1
    │    ├── Subsection 1.1
    │    └── Subsection 1.2
    └── Section 2
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: int = 0                      # 0=document, 1=section, 2=subsection, ...
    title: Optional[str] = None
    text: str = ""                      # raw text of this node
    parent_id: Optional[str] = None
    children: List["DocumentNode"] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add_child(self, child: "DocumentNode"):
        child.parent_id = self.id
        self.children.append(child)
