"""RAG Pipeline orchestrator module with SGR functionality."""

import json
import logging
from dataclasses import dataclass, field
from typing import Iterator

from src.config import settings
from src.data_processing.loader import DocumentLoader
from src.generation import GenerationProvider, create_generator
from src.pipeline.prompts import (
    FinalAnswerPrompt,
    InformationCompletenessPrompt,
    RelevanceCheckPrompt,
)
from src.retrieval import Retriever
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Represents a complete RAG pipeline response."""

    query: str
    generated_document: str
    retrieved_results: list[str]
    additional_questions: list[str] = None
    sources: list[str] = field(default_factory=list)

@dataclass
class IndexingResult:
    """Represents the result of document indexing."""

    documents_processed: int
    total_chunks: int
    status: str


class RAGPipeline:
    """
    Main RAG (Retrieval-Augmented Generation) pipeline orchestrator.

    Coordinates the flow between:
    1. Document loading and chunking
    2. Vector indexing
    3. Semantic retrieval
    4. Document generation

    Implements SGR (Search-Generate-Response) pattern with iterative retrieval:
    1. Search: Retrieve relevant context
    2. Check: Evaluate relevance and completeness
    3. Iterate: Ask clarifying questions and retrieve more if needed
    4. Generate: Create final response
    5. Response: Return structured answer

    Follows SOLID principles:
    - Single Responsibility: Each component has one job
    - Open/Closed: Easy to extend with new providers
    - Dependency Inversion: Depends on abstractions, not concrete classes
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        generator: GenerationProvider | None = None,
        data_dir: str | None = None,
        top_k: int = settings.TOP_K_RESULTS,
        top_k_additional: int = settings.TOP_K_ADDITIONAL,
        min_score: float = settings.DOC_MIN_SCORE,
        max_iterations: int = settings.SGR_MAX_ITERATIONS,
        max_additional_questions: int = settings.SGR_MAX_ADDITIONAL_QUESTIONS,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: VectorStore instance (created if not provided).
            generator: GenerationProvider instance (created if not provided).
            data_dir: Directory containing source documents.
            top_k: Number of results to retrieve initially.
            top_k_additional: Number of results for additional queries.
            max_iterations: Maximum iterations for clarifying questions.
            max_additional_questions: Maximum additional questions per iteration.
        """
        self._vector_store = vector_store or VectorStore()
        self._generator = generator or create_generator()
        self._data_dir = data_dir
        self._retriever = Retriever(
            self._vector_store, 
            top_k=top_k, 
            min_score=min_score,
        )
        self._retriever_additional = Retriever(
            self._vector_store, 
            top_k=top_k_additional,
            min_score=min_score,
        )
        self._max_iterations = max_iterations
        self._max_additional_questions = max_additional_questions

        logger.info(
            f"RAGPipeline initialized with top_k={top_k}, "
            f"max_iterations={max_iterations}"
        )

    def index_documents(self, data_dir: str | None = None) -> IndexingResult:
        """
        Index all documents from the data directory.

        Args:
            data_dir: Optional override for data directory.

        Returns:
            IndexingResult with processing statistics.
        """
        data_dir = data_dir or self._data_dir
        if not data_dir:
            raise ValueError("Data directory not specified")

        logger.info(f"Indexing documents from: {data_dir}")

        loader = DocumentLoader(data_dir)
        documents = list(loader.load_all_documents())

        if not documents:
            return IndexingResult(
                documents_processed=0,
                total_chunks=0,
                status="no_documents_found",
            )

        total_chunks = self._vector_store.index_all_documents(documents)

        result = IndexingResult(
            documents_processed=len(documents),
            total_chunks=total_chunks,
            status="success",
        )

        logger.info(
            f"Indexing complete: {len(documents)} documents, "
            f"{total_chunks} chunks"
        )

        return result

    def query(self, user_query: str) -> RAGResponse:
        """
        Process a user query through the full SGR RAG pipeline.

        SGR Pattern with iterative retrieval:
        1. Search: Retrieve relevant context
        2. Check relevance: Evaluate each document with LLM
        3. Check completeness: Ask LLM if info is sufficient
        4. Iterate: For each clarifying question, retrieve and answer
        5. Generate: Create final response
        6. Response: Return structured answer

        Args:
            user_query: User's query/request.

        Returns:
            RAGResponse with generated document and sources.
        """
        logger.info(f"Processing query with SGR pattern: {user_query}")

        # Step 1: Initial Search (Retrieval)
        retrieved_results = self._retriever.retrieve_formatted(user_query)
        logger.info(f"Initial retrieval: {len(retrieved_results)} results")

        # Step 2: Check relevance of each document using LLM
        relevant_results = self._filter_by_relevance(user_query, retrieved_results)
        logger.info(f"After relevance check: {len(relevant_results)} results")

        if not relevant_results:
            # No relevant documents found
            response = self._generate_no_info_response(user_query)
            return RAGResponse(
                query=user_query,
                generated_document=response,
                retrieved_results=retrieved_results,
                sources=[],
                iterations_count=0,
            )

        # Step 3: Check if information is complete
        additional_questions = self._check_information_completeness(
            query=user_query, 
            contexts="\n\n".join(relevant_results),
        )

        # Step 4: Iterative retrieval for additional questions
        additional_qa_pairs = []
        if additional_questions:
            logger.info(
                f"Information incomplete. Additional questions: {additional_questions}"
            )
            additional_qa_pairs = self._iterative_retrieval(
                additional_questions=additional_questions, 
                existing_results=relevant_results,
            )

        # Step 5: Generate final answer
        final_answer = self._generate_final_answer(
            user_query, relevant_results, additional_qa_pairs
        )

        # Step 6: Response
        response = RAGResponse(
            query=user_query,
            generated_document=final_answer,
            retrieved_results=relevant_results,
            additional_questions=additional_questions,
            sources=relevant_results,
        )

        logger.info(f"Query processed successfully.")
        return response

    def query_stream(
        self, user_query: str
    ) -> Iterator[str]:
        """
        Process a query with streaming generation.

        Args:
            user_query: User's query/request.

        Yields:
            Chunks of the generated document.
        """
        logger.info(f"Processing streaming query: {user_query}")

        # Step 1: Initial Search (Retrieval)
        retrieved_results = self._retriever.retrieve_formatted(user_query)
        logger.info(f"Initial retrieval: {len(retrieved_results)} results")

        # Step 2: Check relevance of each document using LLM
        relevant_results = self._filter_by_relevance(user_query, retrieved_results)
        logger.info(f"After relevance check: {len(relevant_results)} results")

        if not relevant_results:
            yield "Не найдено релевантной информации в базе знаний."
            return

        # Step 3: Format contexts
        contexts = "\n\n".join(relevant_results)

        # Step 4: Check if information is complete
        additional_questions = self._check_information_completeness(
            query=user_query, 
            contexts=contexts,
        )

        # Step 5: Iterative retrieval for additional questions
        additional_qa_pairs = []
        if additional_questions:
            additional_qa_pairs = self._iterative_retrieval(
                additional_questions=additional_questions, 
                existing_results=relevant_results,
            )

        # Step 6: Generate final answer
        messages = [
            {"role": "system", "content": FinalAnswerPrompt.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": FinalAnswerPrompt.build_user_prompt(
                    user_query, contexts, additional_qa_pairs
                ),
            },
        ]

        yield from self._generator.generate_stream(messages)

    def _filter_by_relevance(self, query: str, results: list[str]) -> list[str]:
        """
        Filter results by relevance using LLM.

        Args:
            query: User query.
            results: List of retrieval results.

        Returns:
            List of relevant results.
        """
        relevant_results = []

        for context in results:
            is_relevant = self._check_document_relevance(query, context)
            if is_relevant:
                relevant_results.append(context)

        return relevant_results

    def _check_document_relevance(self, query: str, document: str) -> bool:
        """
        Check if a single document is relevant to the query using LLM.

        Args:
            query: User query.
            result: Retrieval result to check.

        Returns:
            True if document is relevant, False otherwise.
        """
        messages = [
            {"role": "system", "content": RelevanceCheckPrompt.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": RelevanceCheckPrompt.build_user_prompt(query, document),
            },
        ]

        try:
            response = self._generator.generate(messages)
            response_lower = response.strip().lower()
            is_relevant = "yes" in response_lower
            logger.debug(f"Relevance check for '{document[:16]}': {is_relevant}")
            return is_relevant
        except Exception as e:
            logger.warning(f"Relevance check failed: {e}. Assuming relevant.")
            return True

    def _check_information_completeness(
        self, query: str, contexts: str
    ) -> list[str]:
        """
        Check if contexts contain all necessary information.

        Args:
            query: User query.
            contexts: Formatted contexts from retrieval.

        Returns:
            List of additional questions if information is incomplete.
        """
        messages = [
            {
                "role": "system",
                "content": InformationCompletenessPrompt.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": InformationCompletenessPrompt.build_user_prompt(
                    query, contexts
                ),
            },
        ]

        try:
            response = self._generator.generate(messages)
            logger.debug(f"Completeness check response: {response}")

            # Parse JSON response
            response = response[response.find("{") : response.rfind("}") + 1]
            data = json.loads(response)
            questions = data.get("questions", [])

            # Limit number of questions
            questions = questions[: self._max_additional_questions]
            return questions

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Completeness check failed: {e}")
            return []

    def _iterative_retrieval(
        self,
        additional_questions: list[str],
        existing_results: list[str],
    ) -> list[dict]:
        """
        Perform iterative retrieval for additional questions.

        Args:
            additional_questions: List of clarifying questions.
            existing_results: Results from initial retrieval.

        Returns:
            list of Q&A pairs.
        """
        qa_pairs = []
        iterations_count = 0

        found_contexts = set(existing_results)
        for question in additional_questions:
            if iterations_count >= self._max_iterations:
                logger.info("Max iterations reached")
                break

            # Retrieve for this question
            results = self._retriever_additional.retrieve_formatted(question)
            logger.debug(
                f"Retrieved {len(results)} results for: {question}"
            )

            # Filter by relevance
            relevant_results = self._filter_by_relevance(question, results)

            if relevant_results:
                # Generate answer for this question
                messages = [
                    {"role": "system", "content": FinalAnswerPrompt.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": FinalAnswerPrompt.build_user_prompt(
                            user_query=question, 
                            contexts="\n\n".join(relevant_results),
                        ),
                    },
                ]

                answer = self._generator.generate(messages)
                qa_pairs.append({"question": question, "answer": answer})
                iterations_count += 1

                # Add to existing results for final generation
                for context in relevant_results:
                    if context not in found_contexts:
                        existing_results.append(context)
                        found_contexts.add(context)

        return qa_pairs

    def _generate_final_answer(
        self,
        user_query: str,
        contexts: list[str],
        additional_qa_pairs: list[dict],
    ) -> str:
        """
        Generate final answer using all available information.

        Args:
            user_query: Original user query.
            contexts: Relevant contexts.
            additional_qa_pairs: Q&A pairs from iterative retrieval.

        Returns:
            Generated final answer.
        """
        additional_qa = self._format_additional_qa(additional_qa_pairs)

        messages = [
            {"role": "system", "content": FinalAnswerPrompt.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": FinalAnswerPrompt.build_user_prompt(
                    user_query, "\n\n".join(contexts), additional_qa
                ),
            },
        ]

        return self._generator.generate(messages)

    def _generate_no_info_response(self, query: str) -> str:
        """
        Generate response when no relevant information is found.

        Args:
            query: User query.

        Returns:
            Response indicating no information available.
        """
        messages = [
            {"role": "system", "content": FinalAnswerPrompt.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Вопрос пользователя:\n{query}\n\n"
                f"В базе знаний не найдено релевантной информации для ответа на этот вопрос. "
                f"Пожалуйста, уточните запрос или обратитесь к другому источнику информации.",
            },
        ]
        return self._generator.generate(messages)

    def _format_additional_qa(self, qa_pairs: list[dict]) -> str:
        """
        Format additional Q&A pairs for inclusion in final prompt.

        Args:
            qa_pairs: List of Q&A dicts.

        Returns:
            Formatted string.
        """
        if not qa_pairs:
            return ""

        parts = []
        for i, qa in enumerate(qa_pairs, 1):
            parts.append(f"Дополнительный вопрос {i}: {qa['question']}")
            parts.append(f"Ответ: {qa['answer']}")

        return "\n\n".join(parts)

    @property
    def vector_store(self) -> VectorStore:
        """Access the vector store."""
        return self._vector_store

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed chunks."""
        return self._vector_store.count
