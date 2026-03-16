"""Tests for prompts module."""

import pytest
import json

from src.pipeline.prompts import (
    RelevanceCheckPrompt,
    InformationCompletenessPrompt,
    FinalAnswerPrompt,
)


class TestRelevanceCheckPrompt:
    """Tests for RelevanceCheckPrompt class."""

    def test_system_prompt_defined(self):
        """Test that system prompt is defined."""
        assert RelevanceCheckPrompt.SYSTEM_PROMPT is not None
        assert len(RelevanceCheckPrompt.SYSTEM_PROMPT) > 0

    def test_system_prompt_contains_key_instructions(self):
        """Test that system prompt contains key instructions."""
        prompt = RelevanceCheckPrompt.SYSTEM_PROMPT
        assert "yes" in prompt.lower()
        assert "no" in prompt.lower()
        assert "релевантность" in prompt.lower() or "relevant" in prompt.lower()

    def test_build_user_prompt(self):
        """Test building user prompt for relevance check."""
        question = "How to get social services?"
        document = "Social services are available for residents."

        user_prompt = RelevanceCheckPrompt.build_user_prompt(question, document)

        assert question in user_prompt
        assert document in user_prompt
        assert "yes" in user_prompt.lower() or "no" in user_prompt.lower()

    def test_build_user_prompt_format(self):
        """Test user prompt format structure."""
        question = "Test question"
        document = "Test document content"

        user_prompt = RelevanceCheckPrompt.build_user_prompt(question, document)

        assert "Вопрос:" in user_prompt or "Question:" in user_prompt
        assert "Документ:" in user_prompt or "Document:" in user_prompt

    def test_build_user_prompt_with_multiline_document(self):
        """Test building prompt with multiline document."""
        question = "Test question"
        document = """Line 1
        Line 2
        Line 3"""

        user_prompt = RelevanceCheckPrompt.build_user_prompt(question, document)

        assert "Line 1" in user_prompt
        assert "Line 3" in user_prompt


class TestInformationCompletenessPrompt:
    """Tests for InformationCompletenessPrompt class."""

    def test_system_prompt_defined(self):
        """Test that system prompt is defined."""
        assert InformationCompletenessPrompt.SYSTEM_PROMPT is not None
        assert len(InformationCompletenessPrompt.SYSTEM_PROMPT) > 0

    def test_system_prompt_mentions_json_format(self):
        """Test that system prompt mentions JSON format."""
        prompt = InformationCompletenessPrompt.SYSTEM_PROMPT
        assert "json" in prompt.lower()
        assert "questions" in prompt.lower()

    def test_system_prompt_empty_questions_example(self):
        """Test that system prompt shows empty questions example."""
        prompt = InformationCompletenessPrompt.SYSTEM_PROMPT
        assert '{"questions": []}' in prompt

    def test_build_user_prompt(self):
        """Test building user prompt for completeness check."""
        question = "What documents are needed?"
        contexts = "Context 1\nContext 2"

        user_prompt = InformationCompletenessPrompt.build_user_prompt(
            question, contexts
        )

        assert question in user_prompt
        assert contexts in user_prompt
        assert "JSON" in user_prompt or "json" in user_prompt

    def test_build_user_prompt_with_multiple_contexts(self):
        """Test building prompt with multiple contexts."""
        question = "Test question"
        contexts = "Context A\n\nContext B\n\nContext C"

        user_prompt = InformationCompletenessPrompt.build_user_prompt(
            question, contexts
        )

        assert "Context A" in user_prompt
        assert "Context C" in user_prompt

    def test_build_user_prompt_asks_for_json(self):
        """Test that user prompt asks for JSON response."""
        question = "Test"
        contexts = "Some context"

        user_prompt = InformationCompletenessPrompt.build_user_prompt(
            question, contexts
        )

        assert "JSON" in user_prompt or "json" in user_prompt


class TestFinalAnswerPrompt:
    """Tests for FinalAnswerPrompt class."""

    def test_system_prompt_defined(self):
        """Test that system prompt is defined."""
        assert FinalAnswerPrompt.SYSTEM_PROMPT is not None
        assert len(FinalAnswerPrompt.SYSTEM_PROMPT) > 0

    def test_system_prompt_contains_requirements(self):
        """Test that system prompt contains response requirements."""
        prompt = FinalAnswerPrompt.SYSTEM_PROMPT
        assert "контекстов" in prompt.lower() or "context" in prompt.lower()
        assert "структурированным" in prompt.lower() or "structured" in prompt.lower()

    def test_build_user_prompt_basic(self):
        """Test building basic user prompt."""
        query = "How to apply?"
        contexts = "Application context"

        user_prompt = FinalAnswerPrompt.build_user_prompt(query, contexts)

        assert query in user_prompt
        assert contexts in user_prompt

    def test_build_user_prompt_with_additional_qa(self):
        """Test building prompt with additional Q&A pairs."""
        query = "Main question"
        contexts = "Main context"
        additional_qa = "Additional Q1: Question\nAdditional A1: Answer"

        user_prompt = FinalAnswerPrompt.build_user_prompt(
            query, contexts, additional_qa
        )

        assert query in user_prompt
        assert contexts in user_prompt
        assert additional_qa in user_prompt

    def test_build_user_prompt_without_additional_qa(self):
        """Test building prompt without additional Q&A."""
        query = "Question"
        contexts = "Context"

        user_prompt = FinalAnswerPrompt.build_user_prompt(query, contexts)

        assert query in user_prompt
        assert contexts in user_prompt
        assert "Дополнительные вопросы" not in user_prompt

    def test_build_user_prompt_with_empty_additional_qa(self):
        """Test building prompt with empty additional Q&A."""
        query = "Question"
        contexts = "Context"

        user_prompt = FinalAnswerPrompt.build_user_prompt(query, contexts, "")

        assert query in user_prompt
        assert contexts in user_prompt

    def test_build_user_prompt_format_structure(self):
        """Test that prompt has proper structure."""
        query = "Test query"
        contexts = "Test contexts"

        user_prompt = FinalAnswerPrompt.build_user_prompt(query, contexts)

        assert "Вопрос пользователя:" in user_prompt or "user" in user_prompt.lower()
        assert "Контексты" in user_prompt or "context" in user_prompt.lower()

    def test_build_user_prompt_multiple_contexts(self):
        """Test building prompt with multiple contexts."""
        query = "Question"
        contexts = "Context 1\n\nContext 2\n\nContext 3"

        user_prompt = FinalAnswerPrompt.build_user_prompt(query, contexts)

        assert "Context 1" in user_prompt
        assert "Context 3" in user_prompt
