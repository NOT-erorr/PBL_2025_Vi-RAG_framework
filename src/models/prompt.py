from typing import List


class PromptBuilder:
    """
    Default prompt policy for RAG
    """

    def __init__(
        self,
        system_instruction: str | None = None,
        separator: str = "\n\n---\n\n"
    ):
        self.system_instruction = system_instruction or (
            "Bạn là một trợ lý AI trả lời câu hỏi dựa hoàn toàn "
            "trên thông tin được cung cấp."
        )
        self.separator = separator

    def build(
        self,
        query: str,
        contexts: List[str]
    ) -> str:
        context_block = self.separator.join(contexts)

        return f"""
{self.system_instruction}

QUY TẮC:
- Chỉ sử dụng thông tin trong CONTEXT
- Không suy đoán ngoài dữ liệu
- Nếu không đủ thông tin, hãy trả lời rõ ràng

CONTEXT:
{context_block}

CÂU HỎI:
{query}

TRẢ LỜI:
""".strip()
