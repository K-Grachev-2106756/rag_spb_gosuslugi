"""Tests for RAG pipeline module."""

import pytest
from unittest.mock import Mock, patch

from src.pipeline.rag import (
    RAGPipeline,
    RAGResponse,
    IndexingResult,
)
from src.vector_store import VectorStore
from src.generation import GenerationProvider


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""

    def test_rag_response_creation(self):
        """Test creating a RAGResponse instance."""
        response = RAGResponse(
            query="Test query",
            generated_document="Generated doc",
            retrieved_results=["Result 1", "Result 2"],
            additional_questions=["Q1", "Q2"],
            sources=["Source 1"],
        )
        assert response.query == "Test query"
        assert response.generated_document == "Generated doc"
        assert len(response.retrieved_results) == 2
        assert len(response.additional_questions) == 2
        assert len(response.sources) == 1

    def test_rag_response_default_values(self):
        """Test RAGResponse default values."""
        response = RAGResponse(
            query="Query",
            generated_document="Doc",
            retrieved_results=[],
        )
        assert response.additional_questions is None
        assert response.sources == []


class TestIndexingResult:
    """Tests for IndexingResult dataclass."""

    def test_indexing_result_creation(self):
        """Test creating an IndexingResult instance."""
        result = IndexingResult(
            documents_processed=5,
            total_chunks=20,
            status="success",
        )
        assert result.documents_processed == 5
        assert result.total_chunks == 20
        assert result.status == "success"


class TestRAGPipeline:
    """Tests for RAGPipeline class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = Mock(spec=VectorStore)
        store.count = 0
        store.index_all_documents = Mock(return_value=10)
        store.search = Mock(return_value=[])
        return store

    @pytest.fixture
    def mock_generator(self):
        """Create a mock generator."""
        generator = Mock(spec=GenerationProvider)
        generator.generate = Mock(return_value="Generated response")
        
        # Create async generator for generate_stream
        async def async_gen(*args, **kwargs):
            for chunk in ["Chunk 1", "Chunk 2"]:
                yield chunk
        
        generator.generate_stream = async_gen
        return generator

    def test_pipeline_initialization(self, mock_vector_store, mock_generator):
        """Test pipeline initializes correctly."""
        from src.config import settings

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
            data_dir="/test/data",
        )
        assert pipeline._data_dir == "/test/data"
        assert pipeline._vector_store == mock_vector_store
        assert pipeline._generator == mock_generator

    def test_pipeline_default_initialization(self, mock_vector_store, mock_generator):
        """Test pipeline with default parameters."""
        from src.config import settings

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )
        # Check that retrievers are created with settings values
        assert pipeline._retriever._top_k == settings.TOP_K_RESULTS
        assert pipeline._max_iterations == settings.SGR_MAX_ITERATIONS

    def test_index_documents_success(self, mock_vector_store, mock_generator):
        """Test successful document indexing."""
        from src.data_processing.loader import Document

        # Mock document loader
        mock_doc = Document(
            title="Test Doc",
            description_part="Description",
            info_part=[("Section", "Content")],
            source_path="test.html",
        )

        with patch('src.pipeline.rag.DocumentLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load_all_documents = Mock(return_value=[mock_doc])
            mock_loader_class.return_value = mock_loader

            pipeline = RAGPipeline(
                vector_store=mock_vector_store,
                generator=mock_generator,
                data_dir="/test/data",
            )
            result = pipeline.index_documents()

            assert result.documents_processed == 1
            assert result.total_chunks == 10
            assert result.status == "success"
            mock_vector_store.index_all_documents.assert_called_once()

    def test_index_documents_no_data_dir(self, mock_vector_store, mock_generator):
        """Test indexing without data directory raises error."""
        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        with pytest.raises(ValueError, match="Data directory not specified"):
            pipeline.index_documents()

    def test_index_documents_override_data_dir(self, mock_vector_store, mock_generator):
        """Test indexing with overridden data directory."""
        from src.data_processing.loader import Document

        mock_doc = Document(
            title="Test",
            description_part="Desc",
            info_part=[],
            source_path="test.html",
        )

        with patch('src.pipeline.rag.DocumentLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load_all_documents = Mock(return_value=[mock_doc])
            mock_loader_class.return_value = mock_loader

            pipeline = RAGPipeline(
                vector_store=mock_vector_store,
                generator=mock_generator,
                data_dir="/original/data",
            )
            result = pipeline.index_documents(data_dir="/override/data")

            mock_loader_class.assert_called_once_with("/override/data")

    def test_index_documents_empty_directory(self, mock_vector_store, mock_generator):
        """Test indexing when no documents found."""
        with patch('src.pipeline.rag.DocumentLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load_all_documents = Mock(return_value=[])
            mock_loader_class.return_value = mock_loader

            pipeline = RAGPipeline(
                vector_store=mock_vector_store,
                generator=mock_generator,
                data_dir="/test/data",
            )
            result = pipeline.index_documents()

            assert result.documents_processed == 0
            assert result.total_chunks == 0
            assert result.status == "no_documents_found"

    def test_query_basic(self, mock_vector_store, mock_generator):
        """Test basic query processing."""
        mock_vector_store.search = Mock(return_value=[
            {
                "content": "Relevant content",
                "metadata": {"document_title": "Doc", "metadata": "Meta", "parent_section": ""},
                "distance": 0.2,
            }
        ])

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
            data_dir="/test/data",
        )

        # Mock the relevance check to return True
        with patch.object(pipeline, '_check_document_relevance', return_value=True):
            # Mock completeness check to return empty (no additional questions)
            with patch.object(pipeline, '_check_information_completeness', return_value=[]):
                response = pipeline.query("Test query")

                assert isinstance(response, RAGResponse)
                assert response.query == "Test query"
                assert response.generated_document == "Generated response"

    def test_query_no_relevant_results(self, mock_vector_store, mock_generator):
        """Test query when no relevant results found."""
        mock_vector_store.search = Mock(return_value=[])

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
            data_dir="/test/data",
        )

        response = pipeline.query("Test query")

        assert isinstance(response, RAGResponse)
        assert response.query == "Test query"
        assert response.sources == []

    def test_query_with_additional_questions(self, mock_vector_store, mock_generator):
        """Test query with additional clarifying questions."""
        mock_vector_store.search = Mock(return_value=[
            {
                "content": "Content",
                "metadata": {"document_title": "Doc", "metadata": "Meta", "parent_section": ""},
                "distance": 0.2,
            }
        ])

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
            data_dir="/test/data",
        )

        with patch.object(pipeline, '_check_document_relevance', return_value=True):
            with patch.object(pipeline, '_check_information_completeness', return_value=["Q1"]):
                with patch.object(pipeline, '_iterative_retrieval', return_value=[{"question": "Q1", "answer": "A1"}]):
                    response = pipeline.query("Test query")

                    assert isinstance(response, RAGResponse)
                    assert response.additional_questions == ["Q1"]

    @pytest.mark.asyncio
    async def test_query_stream(self, mock_vector_store, mock_generator):
        """Test streaming query processing."""
        mock_vector_store.search = Mock(return_value=[
            {
                "content": "Content",
                "metadata": {"document_title": "Doc", "metadata": "Meta", "parent_section": ""},
                "distance": 0.2,
            }
        ])

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
            data_dir="/test/data",
        )

        with patch.object(pipeline, '_check_document_relevance', return_value=True):
            with patch.object(pipeline, '_check_information_completeness', return_value=[]):
                chunks = [chunk async for chunk in pipeline.query_stream("Test query")]

                assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_query_stream_no_results(self, mock_vector_store, mock_generator):
        """Test streaming query with no results."""
        mock_vector_store.search = Mock(return_value=[])

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
            data_dir="/test/data",
        )

        chunks = [chunk async for chunk in pipeline.query_stream("Test query")]

        assert len(chunks) == 1
        assert "Не найдено релевантной информации" in chunks[0]

    def test_check_document_relevance_yes(self, mock_vector_store, mock_generator):
        """Test relevance check returning yes."""
        mock_generator.generate = Mock(return_value="yes")

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        is_relevant = pipeline._check_document_relevance("Query", "Document")

        assert is_relevant is True
        mock_generator.generate.assert_called_once()

    def test_check_document_relevance_no(self, mock_vector_store, mock_generator):
        """Test relevance check returning no."""
        mock_generator.generate = Mock(return_value="no")

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        is_relevant = pipeline._check_document_relevance("Query", "Document")

        assert is_relevant is False

    def test_check_document_relevance_error(self, mock_vector_store, mock_generator):
        """Test relevance check handles errors gracefully."""
        mock_generator.generate = Mock(side_effect=Exception("API Error"))

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        is_relevant = pipeline._check_document_relevance("Query", "Document")

        assert is_relevant is True  # Defaults to True on error

    def test_check_information_completeness(self, mock_vector_store, mock_generator):
        """Test information completeness check."""
        mock_generator.generate = Mock(return_value='{"questions": ["Q1", "Q2"]}')

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        questions = pipeline._check_information_completeness("Query", "Contexts")

        assert questions == ["Q1", "Q2"]

    def test_check_information_completeness_empty(self, mock_vector_store, mock_generator):
        """Test completeness check when info is complete."""
        mock_generator.generate = Mock(return_value='{"questions": []}')

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        questions = pipeline._check_information_completeness("Query", "Contexts")

        assert questions == []

    def test_check_information_completeness_invalid_json(self, mock_vector_store, mock_generator):
        """Test completeness check handles invalid JSON."""
        mock_generator.generate = Mock(return_value='invalid json')

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        questions = pipeline._check_information_completeness("Query", "Contexts")

        assert questions == []

    def test_check_information_completeness_limits_questions(self, mock_vector_store, mock_generator):
        """Test that questions are limited to max_additional_questions."""
        from src.config import settings

        mock_generator.generate = Mock(return_value='{"questions": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]}')

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
            max_additional_questions=3,
        )

        questions = pipeline._check_information_completeness("Query", "Contexts")

        assert len(questions) <= 3

    def test_iterative_retrieval(self, mock_vector_store, mock_generator):
        """Test iterative retrieval for additional questions."""
        mock_vector_store.search = Mock(return_value=[
            {
                "content": "Additional content",
                "metadata": {"document_title": "Doc", "metadata": "Meta", "parent_section": ""},
                "distance": 0.2,
            }
        ])

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        with patch.object(pipeline, '_filter_by_relevance', return_value=["Relevant"]):
            qa_pairs = pipeline._iterative_retrieval(
                additional_questions=["Q1"],
                existing_results=[],
            )

            assert len(qa_pairs) > 0
            assert "question" in qa_pairs[0]
            assert "answer" in qa_pairs[0]

    def test_iterative_retrieval_respects_max_iterations(self, mock_vector_store, mock_generator):
        """Test that iterative retrieval respects max iterations."""
        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
            max_iterations=2,
        )

        with patch.object(pipeline, '_filter_by_relevance', return_value=[]):
            qa_pairs = pipeline._iterative_retrieval(
                additional_questions=["Q1", "Q2", "Q3", "Q4", "Q5"],
                existing_results=[],
            )

            # Should stop at max_iterations
            assert len(qa_pairs) <= 2

    def test_generate_final_answer(self, mock_vector_store, mock_generator):
        """Test final answer generation."""
        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        answer = pipeline._generate_final_answer(
            user_query="Test query",
            contexts=["Context 1", "Context 2"],
            additional_qa_pairs=[],
        )

        assert answer == "Generated response"
        mock_generator.generate.assert_called_once()

    def test_generate_final_answer_with_additional_qa(self, mock_vector_store, mock_generator):
        """Test final answer generation with additional Q&A."""
        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        answer = pipeline._generate_final_answer(
            user_query="Test query",
            contexts=["Context"],
            additional_qa_pairs=[{"question": "Q1", "answer": "A1"}],
        )

        assert answer == "Generated response"

    def test_generate_no_info_response(self, mock_vector_store, mock_generator):
        """Test generating response when no info found."""
        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        response = pipeline._generate_no_info_response("Test query")

        assert response == "Generated response"
        assert "не найдено релевантной информации" in mock_generator.generate.call_args[0][0][1]["content"].lower()

    def test_format_additional_qa(self, mock_vector_store, mock_generator):
        """Test formatting additional Q&A pairs."""
        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        qa_pairs = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]

        formatted = pipeline._format_additional_qa(qa_pairs)

        assert "Дополнительный вопрос 1" in formatted
        assert "Q1" in formatted
        assert "A1" in formatted
        assert "Дополнительный вопрос 2" in formatted

    def test_format_additional_qa_empty(self, mock_vector_store, mock_generator):
        """Test formatting empty Q&A pairs."""
        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        formatted = pipeline._format_additional_qa([])

        assert formatted == ""

    def test_vector_store_property(self, mock_vector_store, mock_generator):
        """Test vector_store property access."""
        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        assert pipeline.vector_store == mock_vector_store

    def test_chunk_count_property(self, mock_vector_store, mock_generator):
        """Test chunk_count property."""
        mock_vector_store.count = 42

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        assert pipeline.chunk_count == 42

    def test_filter_by_relevance(self, mock_vector_store, mock_generator):
        """Test filtering results by relevance."""
        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

        results = ["Doc 1", "Doc 2", "Doc 3"]

        with patch.object(pipeline, '_check_document_relevance', side_effect=[True, False, True]):
            filtered = pipeline._filter_by_relevance("Query", results)

            assert len(filtered) == 2
            assert "Doc 1" in filtered
            assert "Doc 3" in filtered
            assert "Doc 2" not in filtered
