"""Tests for knowledge base tools: prometheus/tools/knowledge.py.

Tests:
- Topic sanitization (validation, reserved names, path traversal)
- Read/write/list operations
- Index management
- Error handling

Run: python -m pytest tests/test_knowledge.py -v
"""
import pathlib
import tempfile

import pytest

from prometheus.tools.knowledge import (
    _sanitize_topic,
    _safe_path,
    _extract_summary,
    _rebuild_index,
    _update_index_entry,
    _knowledge_read,
    _knowledge_write,
    _knowledge_list,
)
from prometheus.tools.registry import ToolContext


# --- Fixtures ---


@pytest.fixture
def tmp_drive():
    """Temporary drive root for testing."""
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "memory" / "knowledge").mkdir(parents=True, exist_ok=True)
    return tmp


@pytest.fixture
def ctx(tmp_drive):
    """ToolContext for testing."""
    return ToolContext(
        repo_dir=tmp_drive,
        drive_root=tmp_drive,
        branch_dev="main",
        pending_events=[],
        current_chat_id=None,
        current_task_type=None,
        emit_progress_fn=None,
        task_depth=0,
        is_direct_chat=False,
    )


# --- Topic sanitization tests ---


class TestSanitizeTopic:
    """Tests for _sanitize_topic function."""

    def test_valid_simple_topic(self):
        """Simple alphanumeric topic passes."""
        assert _sanitize_topic("bitcoin") == "bitcoin"

    def test_valid_topic_with_underscore(self):
        """Topic with underscore passes."""
        assert _sanitize_topic("crypto_market") == "crypto_market"

    def test_valid_topic_with_hyphen(self):
        """Topic with hyphen passes."""
        assert _sanitize_topic("browser-automation") == "browser-automation"

    def test_valid_topic_with_dot(self):
        """Topic with dot passes."""
        assert _sanitize_topic("tech.awareness") == "tech.awareness"

    def test_valid_topic_mixed(self):
        """Mixed alphanumeric topic passes."""
        assert _sanitize_topic("BTC-2024_analysis") == "BTC-2024_analysis"

    def test_rejects_empty_topic(self):
        """Empty topic raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty"):
            _sanitize_topic("")

    def test_rejects_whitespace_only(self):
        """Whitespace-only topic raises ValueError (strips whitespace first, then rejects)."""
        with pytest.raises(ValueError, match="Invalid topic name"):
            _sanitize_topic("   ")

    def test_rejects_path_separator(self):
        """Topic with path separator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid characters"):
            _sanitize_topic("topic/subtopic")

    def test_rejects_backslash(self):
        """Topic with backslash raises ValueError."""
        with pytest.raises(ValueError, match="Invalid characters"):
            _sanitize_topic("topic\\subtopic")

    def test_rejects_path_traversal(self):
        """Path traversal raises ValueError."""
        with pytest.raises(ValueError, match="Invalid characters"):
            _sanitize_topic("../../../etc/passwd")

    def test_rejects_double_dot(self):
        """Double dot in topic raises ValueError."""
        with pytest.raises(ValueError, match="Invalid characters"):
            _sanitize_topic("topic..bad")

    def test_rejects_leading_hyphen(self):
        """Leading hyphen raises ValueError."""
        with pytest.raises(ValueError, match="Invalid topic name"):
            _sanitize_topic("-topic")

    def test_rejects_trailing_hyphen(self):
        """Trailing hyphen raises ValueError."""
        with pytest.raises(ValueError, match="Invalid topic name"):
            _sanitize_topic("topic-")

    def test_rejects_leading_dot(self):
        """Leading dot raises ValueError."""
        with pytest.raises(ValueError, match="Invalid topic name"):
            _sanitize_topic(".topic")

    def test_rejects_reserved_name_con(self):
        """Reserved name 'con' raises ValueError."""
        with pytest.raises(ValueError, match="Reserved topic name"):
            _sanitize_topic("con")

    def test_rejects_reserved_name_prn(self):
        """Reserved name 'prn' raises ValueError."""
        with pytest.raises(ValueError, match="Reserved topic name"):
            _sanitize_topic("prn")

    def test_rejects_reserved_name_aux(self):
        """Reserved name 'aux' raises ValueError."""
        with pytest.raises(ValueError, match="Reserved topic name"):
            _sanitize_topic("aux")

    def test_rejects_reserved_name_nul(self):
        """Reserved name 'nul' raises ValueError."""
        with pytest.raises(ValueError, match="Reserved topic name"):
            _sanitize_topic("nul")

    def test_rejects_reserved_name_prn_uppercase(self):
        """Reserved name 'PRN' (case-insensitive) raises ValueError."""
        with pytest.raises(ValueError, match="Reserved topic name"):
            _sanitize_topic("PRN")

    def test_single_character_topic_valid(self):
        """Single character topic is valid."""
        assert _sanitize_topic("a") == "a"
        assert _sanitize_topic("1") == "1"

    def test_strips_whitespace(self):
        """Whitespace is stripped from topic."""
        assert _sanitize_topic("  topic  ") == "topic"
        assert _sanitize_topic("\ttopic\t") == "topic"


class TestSafePath:
    """Tests for _safe_path function."""

    def test_valid_topic_returns_path(self, ctx, tmp_drive):
        """Valid topic returns path within knowledge directory."""
        path, sanitized = _safe_path(ctx, "bitcoin")
        assert sanitized == "bitcoin"
        assert "memory/knowledge/bitcoin.md" in str(path)

    def test_invalid_topic_raises(self, ctx):
        """Invalid topic raises ValueError."""
        with pytest.raises(ValueError):
            _safe_path(ctx, "../etc/passwd")

    def test_path_traversal_rejected(self, ctx, tmp_drive):
        """Path traversal attempt is caught at sanitization."""
        with pytest.raises(ValueError, match="Invalid characters"):
            _safe_path(ctx, "..")

    def test_path_escape_protection(self, ctx, tmp_drive):
        """_safe_path properly resolves and contains paths."""
        # Even if sanitization passes, path containment is verified
        path, sanitized = _safe_path(ctx, "normal_topic")
        assert "memory/knowledge" in str(path)
        # Verify the path is within the expected directory
        assert path.resolve().parent == (tmp_drive / "memory" / "knowledge").resolve()


class TestExtractSummary:
    """Tests for _extract_summary function."""

    def test_extracts_plain_text(self):
        """Plain text lines are extracted."""
        text = "First line\nSecond line\nThird line"
        summary = _extract_summary(text)
        assert "First line" in summary

    def test_skips_headings(self):
        """Heading lines (#) are skipped."""
        text = "# Title\nSome content here"
        summary = _extract_summary(text)
        assert "Title" not in summary
        assert "Some content" in summary

    def test_strips_markdown_list_markers(self):
        """Markdown list markers are stripped."""
        text = "- First item\n- Second item"
        summary = _extract_summary(text)
        assert "- " not in summary
        assert "First item" in summary

    def test_strips_leading_bold_markers(self):
        """Leading markdown bold markers are stripped."""
        text = "**Bold text** and normal"
        summary = _extract_summary(text)
        assert "**" not in summary.split()[0]  # First snippet shouldn't have leading **
        assert "Bold text" in summary

    def test_limits_to_three_snippets(self):
        """Summary is limited to 3 snippets."""
        text = "One\nTwo\nThree\nFour\nFive"
        summary = _extract_summary(text)
        # Should have at most 3 " | " separators
        assert summary.count(" | ") <= 2

    def test_respects_max_chars(self):
        """Summary is capped at max_chars."""
        long_text = "A" * 500 + "\n" + "B" * 500
        summary = _extract_summary(long_text, max_chars=100)
        assert len(summary) <= 101  # 100 + 1 for ellipsis

    def test_empty_content_returns_empty(self):
        """Empty content returns empty string."""
        assert _extract_summary("") == ""
        assert _extract_summary("   ") == ""

    def test_only_headings_returns_empty(self):
        """Content with only headings returns empty string."""
        text = "# Title\n## Subtitle\n### Section"
        assert _extract_summary(text) == ""


class TestKnowledgeWrite:
    """Tests for _knowledge_write function."""

    def test_write_creates_file(self, ctx, tmp_drive):
        """Writing to topic creates the file."""
        result = _knowledge_write(ctx, "test_topic", "Test content")
        assert "test_topic" in result
        assert "saved" in result.lower()
        
        # Verify file exists
        file_path = tmp_drive / "memory" / "knowledge" / "test_topic.md"
        assert file_path.exists()
        assert file_path.read_text() == "Test content"

    def test_write_overwrite_mode(self, ctx, tmp_drive):
        """Overwrite mode replaces content."""
        _knowledge_write(ctx, "topic", "First content")
        result = _knowledge_write(ctx, "topic", "New content", mode="overwrite")
        
        file_path = tmp_drive / "memory" / "knowledge" / "topic.md"
        assert file_path.read_text() == "New content"

    def test_write_append_mode(self, ctx, tmp_drive):
        """Append mode adds to existing content."""
        _knowledge_write(ctx, "topic", "First\n")
        _knowledge_write(ctx, "topic", "Second", mode="append")
        
        file_path = tmp_drive / "memory" / "knowledge" / "topic.md"
        content = file_path.read_text()
        assert "First" in content
        assert "Second" in content

    def test_write_invalid_mode(self, ctx):
        """Invalid mode returns error."""
        result = _knowledge_write(ctx, "topic", "content", mode="invalid")
        assert "Invalid mode" in result

    def test_write_invalid_topic(self, ctx):
        """Invalid topic returns error."""
        result = _knowledge_write(ctx, "../bad", "content")
        assert "Invalid topic" in result


class TestKnowledgeRead:
    """Tests for _knowledge_read function."""

    def test_read_existing_topic(self, ctx, tmp_drive):
        """Reading existing topic returns content."""
        # Create the file first
        file_path = tmp_drive / "memory" / "knowledge" / "topic.md"
        file_path.write_text("Stored content")
        
        result = _knowledge_read(ctx, "topic")
        assert result == "Stored content"

    def test_read_nonexistent_topic(self, ctx):
        """Reading non-existent topic returns error message."""
        result = _knowledge_read(ctx, "nonexistent")
        assert "not found" in result.lower()

    def test_read_invalid_topic(self, ctx):
        """Reading invalid topic returns error."""
        result = _knowledge_read(ctx, "../bad")
        assert "Invalid topic" in result


class TestKnowledgeList:
    """Tests for _knowledge_list function."""

    def test_list_empty_knowledge_base(self, ctx, tmp_drive):
        """Listing empty knowledge base returns appropriate message."""
        result = _knowledge_list(ctx)
        assert "empty" in result.lower()

    def test_list_with_topics(self, ctx, tmp_drive):
        """Listing with topics shows them."""
        # Create some topics
        _knowledge_write(ctx, "topic1", "Content 1")
        _knowledge_write(ctx, "topic2", "Content 2")
        
        result = _knowledge_list(ctx)
        assert "topic1" in result
        assert "topic2" in result
        assert "Knowledge Base Index" in result

    def test_list_shows_summaries(self, ctx, tmp_drive):
        """List shows topic summaries."""
        _knowledge_write(ctx, "bitcoin", "BTC is a cryptocurrency")
        
        result = _knowledge_list(ctx)
        assert "bitcoin" in result
        assert "BTC" in result


class TestIndexManagement:
    """Tests for index rebuild and update functions."""

    def test_rebuild_index_creates_index(self, ctx, tmp_drive):
        """Rebuild index creates index file."""
        # Create some topics
        _knowledge_write(ctx, "topic1", "Content 1")
        _knowledge_write(ctx, "topic2", "Content 2")
        
        # Rebuild index
        _rebuild_index(ctx)
        
        index_path = tmp_drive / "memory" / "knowledge" / "_index.md"
        assert index_path.exists()
        
        content = index_path.read_text()
        assert "topic1" in content
        assert "topic2" in content

    def test_update_index_entry(self, ctx, tmp_drive):
        """Updating index entry works correctly."""
        # Create topic file directly
        file_path = tmp_drive / "memory" / "knowledge" / "test.md"
        file_path.write_text("Test content for indexing")
        
        # Update index for this topic
        _update_index_entry(ctx, "test")
        
        index_path = tmp_drive / "memory" / "knowledge" / "_index.md"
        assert index_path.exists()
        
        content = index_path.read_text()
        assert "test" in content

    def test_index_maintains_sorted_order(self, ctx, tmp_drive):
        """Index entries are maintained in sorted order."""
        # Create topics in non-alphabetical order
        _knowledge_write(ctx, "zebra", "Z content")
        _knowledge_write(ctx, "apple", "A content")
        _knowledge_write(ctx, "banana", "B content")
        
        result = _knowledge_list(ctx)
        
        # Check order (should be alphabetical)
        zebra_pos = result.find("zebra")
        apple_pos = result.find("apple")
        banana_pos = result.find("banana")
        
        assert apple_pos < banana_pos < zebra_pos
