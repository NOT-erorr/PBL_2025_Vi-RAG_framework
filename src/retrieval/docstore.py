from typing import Protocol, Dict, Any, Iterable, Optional


class DocStore(Protocol):
    def add(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        ...

    def get(self, doc_id: str) -> Optional[str]:
        ...

    def get_with_metadata(
        self,
        doc_id: str
    ) -> Optional[Dict[str, Any]]:
        ...

    def bulk_add(
        self,
        docs: Iterable[Dict[str, Any]]
    ) -> None:
        ...


# Helper functions for parent document retrieval
def fetch_parent_contexts(
    docstore,
    parent_ids: list[str],
    max_docs: int | None = None
) -> list[str]:
    """
    Fetch parent document contexts from docstore.
    Deduplicate parent_ids while preserving order.
    """
    seen = set()
    contexts = []

    for pid in parent_ids:
        if pid in seen:
            continue
        seen.add(pid)

        text = docstore.get(pid)
        if text:
            contexts.append(text)

        if max_docs and len(contexts) >= max_docs:
            break

    return contexts


def fetch_parent_records(
    docstore,
    parent_ids: list[str]
) -> list[dict]:
    """
    Fetch parent document records with metadata from docstore.
    """
    records = []

    for pid in dict.fromkeys(parent_ids):
        record = docstore.get_with_metadata(pid)
        if record:
            records.append(record)

    return records
