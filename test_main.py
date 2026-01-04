"""
Test suite for main.py compliance auditor improvements
Tests Neo4j query improvements and source citation validation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    query_neo4j_requirements,
    create_openai_response,
    ChatCompletionRequest,
    Message
)


class TestNeo4jQueryImprovements:
    """Test improvements to query_neo4j_requirements function"""
    
    @patch('main.neo4j_driver', None)
    def test_query_without_driver_returns_empty_and_logs_warning(self, caplog):
        """Test that querying without a driver logs a warning and returns empty string"""
        result = query_neo4j_requirements("test query")
        
        assert result == ""
        assert "Neo4j driver not available" in caplog.text
        assert "⚠️" in caplog.text
    
    @patch('main.neo4j_driver')
    @patch('main.logger')
    def test_query_extracts_keywords(self, mock_logger, mock_driver):
        """Test that query extracts and logs keywords"""
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = []
        
        query_neo4j_requirements("Dies ist ein Test für DORA Compliance")
        
        # Check that logger.info was called with the keywords message
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("🔍 Neo4j search with keywords:" in str(call) for call in info_calls)
    
    @patch('main.neo4j_driver')
    @patch('main.logger')
    def test_query_logs_found_requirements(self, mock_logger, mock_driver):
        """Test that found requirements are logged"""
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # Mock requirements result
        mock_record = {
            'id': 'REQ-001',
            'text': 'Test requirement text',
            'source': 'requirements.pdf',
            'line': 42
        }
        mock_session.run.return_value = [mock_record]
        
        result = query_neo4j_requirements("test")
        
        # Check that logger.info was called with appropriate messages
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("Found" in str(call) and "requirements in Neo4j" in str(call) for call in info_calls)
        assert any("📊 Neo4j query complete:" in str(call) for call in info_calls)
        assert "📄 Quelle:" in result
    
    @patch('main.neo4j_driver')
    def test_query_warns_when_no_results(self, mock_driver, caplog):
        """Test that a warning is logged when no results are found"""
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = []
        
        result = query_neo4j_requirements("nonexistent query")
        
        assert result == ""
        assert "⚠️ Neo4j query returned NO results" in caplog.text
        assert "check if database is populated" in caplog.text
    
    @patch('main.neo4j_driver')
    def test_query_handles_requirements_error_gracefully(self, mock_driver, caplog):
        """Test that errors in requirements query are handled gracefully"""
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = Exception("Database error")
        
        result = query_neo4j_requirements("test")
        
        assert "❌ Error querying requirements:" in caplog.text
        # Should still return (empty) result and not crash
        assert isinstance(result, str)
    
    @patch('main.neo4j_driver')
    def test_query_includes_emoji_markers(self, mock_driver):
        """Test that context includes emoji markers for better readability"""
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # Mock requirements with source
        mock_req_record = {
            'id': 'REQ-001',
            'text': 'Test requirement',
            'source': 'requirements.pdf',
            'line': 10
        }
        
        # Mock EUR-Lex article
        mock_eurlex_record = {
            'number': '5',
            'title': 'Test Article',
            'text': 'Article text',
            'celex': '32022R2554',
            'doc_title': 'DORA Regulation'
        }
        
        mock_session.run.side_effect = [
            [mock_req_record],  # First call for requirements
            [mock_eurlex_record]  # Second call for EUR-Lex
        ]
        
        result = query_neo4j_requirements("test")
        
        assert "📄 Quelle:" in result  # Requirements marker
        assert "📋 Quelle:" in result  # EUR-Lex marker
        assert "CELEX:" in result


class TestChatCompletionImprovements:
    """Test improvements to chat completion endpoint"""
    
    def test_create_openai_response_format(self):
        """Test that OpenAI response format is correct"""
        response = create_openai_response("Test content", "test-model")
        
        assert response.object == "chat.completion"
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0]["message"]["content"] == "Test content"
        assert response.choices[0]["message"]["role"] == "assistant"
    
    def test_chat_request_model_validation(self):
        """Test ChatCompletionRequest model validation"""
        request = ChatCompletionRequest(
            model="llama2",
            messages=[
                Message(role="user", content="Test message")
            ],
            stream=False
        )
        
        assert request.model == "llama2"
        assert len(request.messages) == 1
        assert request.stream is False
        assert request.temperature == 0.7  # default
        assert request.max_tokens == 2000  # default


class TestSourceCitationValidation:
    """Test source citation validation logic"""
    
    def test_detect_celex_citation(self):
        """Test detection of CELEX citations"""
        response = "Gemäß [CELEX:32022R2554, Artikel 5] müssen..."
        
        has_celex = "[CELEX:" in response or "CELEX:" in response
        assert has_celex is True
    
    def test_detect_pdf_citation(self):
        """Test detection of PDF citations"""
        response = "Laut [Policy_IKT.pdf, Seite 3] ist..."
        
        has_pdf = ".pdf" in response
        assert has_pdf is True
    
    def test_detect_requirement_citation(self):
        """Test detection of requirement citations"""
        response = "Die [SOLL-ANFORDERUNG: REQ-IKT-042] fordert..."
        
        has_req = "[REQ" in response or "[SOLL-ANFORDERUNG" in response
        assert has_req is True
    
    def test_detect_missing_citations(self):
        """Test detection when no citations are present"""
        response = "Dies ist eine Antwort ohne Quellen."
        
        has_celex = "[CELEX:" in response or "CELEX:" in response
        has_pdf = ".pdf" in response
        has_req = "[REQ" in response or "[SOLL-ANFORDERUNG" in response
        
        assert not (has_celex or has_pdf or has_req)


class TestSourceExtraction:
    """Test source extraction from Neo4j context"""
    
    def test_extract_sources_from_context(self):
        """Test extraction of sources from formatted context"""
        import re
        
        context = """
        📄 Quelle: requirements.pdf, Zeile 42
        [SOLL-ANFORDERUNG: REQ-001]
        Test requirement text
        
        📋 Quelle: CELEX:32022R2554, Artikel 5
        Dokument: DORA Regulation
        Test article text
        """
        
        sources = re.findall(r'(?:📄|📋)\s*Quelle:\s*([^\n]+)', context)
        
        assert len(sources) == 2
        assert "requirements.pdf, Zeile 42" in sources
        assert "CELEX:32022R2554, Artikel 5" in sources


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
