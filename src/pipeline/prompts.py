"""Prompts for SGR RAG pipeline."""


class RelevanceCheckPrompt:
    """Prompt for checking if a document chunk contains relevant information."""

    SYSTEM_PROMPT = """Вы - главный ассистент портала "Государственные услуги Санкт-Петербурга", который оценивает релевантность документа заданному вопросу.
Ваша задача - определить, содержит ли предоставленный документ полезную информацию для ответа на вопрос.

Ответьте ТОЛЬКО "yes" или "no":
- "yes" - если в документе есть информация, которая может помочь ответить на вопрос
- "no" - если документ не содержит полезной информации для ответа на вопрос"""

    @staticmethod
    def build_user_prompt(question: str, document_content: str) -> str:
        """Build user prompt for relevance check."""
        return f"""Вопрос: {question}

Документ:
{document_content}

Содержит ли этот документ полезную информацию для ответа на вопрос? (ответьте только "yes" или "no")"""


class InformationCompletenessPrompt:
    """Prompt for checking if all necessary information is available."""

    SYSTEM_PROMPT = """Вы - главный ассистент портала "Государственные услуги Санкт-Петербурга", который оценивает полноту информации для ответа на вопрос.
Ваша задача - определить, достаточно ли предоставленных контекстов для полного ответа на вопрос пользователя.

Если информации достаточно, ответьте:
{"questions": []}

Если информации недостаточно, ответьте в формате JSON со списком уточняющих вопросов:
{"questions": ["вопрос 1", "вопрос 2", ...]}

Важно:
- Задавайте только конкретные уточняющие вопросы, которые помогут найти недостающую информацию об основых правилах и положениях документооборота в рамках портала "Государственные услуги Санкт-Петербурга"
- Не повторяйте исходный вопрос пользователя
- Ответ должен быть ТОЛЬКО в формате JSON"""

    @staticmethod
    def build_user_prompt(
        question: str, contexts: str
    ) -> str:
        """Build user prompt for information completeness check."""
        return f"""Вопрос пользователя:
{question}

Доступные контексты:
{contexts}

Достаточно ли этой информации для полного ответа на вопрос? Если нет, какие уточняющие вопросы нужно задать?
Ответьте в формате JSON: {{"questions": [...]}}"""


class FinalAnswerPrompt:
    """Prompt for generating the final answer."""

    SYSTEM_PROMPT = """Вы - профессиональный ассистент, который отвечает на вопросы на основе предоставленных контекстов.

Требования к ответу:
- Используйте ТОЛЬКО информацию из предоставленных контекстов
- Не выдумывайте факты
- Если информации недостаточно, укажите это
- Ответ должен быть полным, структурированным и полезным
- Официально-деловой стиль изложения"""

    @staticmethod
    def build_user_prompt(
        user_query: str,
        contexts: str,
        additional_qa: str = "",
    ) -> str:
        """
        Build user prompt for final answer generation.

        Args:
            user_query: Original user query.
            contexts: Retrieved contexts from initial search.
            additional_qa: Optional additional Q&A pairs from iterative retrieval.

        Returns:
            Formatted prompt string.
        """
        prompt_parts = [f"Вопрос пользователя:\n{user_query}"]

        if additional_qa:
            prompt_parts.append(f"\nДополнительные вопросы и ответы:\n{additional_qa}")

        prompt_parts.append(f"\nКонтексты из базы знаний:\n{contexts}")

        prompt_parts.append(
            "\nНа основе приведённой информации дайте полный и структурированный ответ на вопрос пользователя."
        )

        return "\n".join(prompt_parts)
