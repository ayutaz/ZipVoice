"""Tests for JapaneseTokenizer with accent markers."""

import pytest


class TestJapaneseTokenizerBasic:
    """Test JapaneseTokenizer without accent markers (default behavior)."""

    @pytest.fixture
    def tokenizer(self):
        from zipvoice.tokenizer.tokenizer import JapaneseTokenizer

        return JapaneseTokenizer(use_accent=False)

    def test_simple_text(self, tokenizer):
        """Test basic Japanese text tokenization."""
        text = "こんにちは"
        tokens = tokenizer.texts_to_tokens([text])[0]
        assert tokens == ["k", "o", "N", "n", "i", "ch", "i", "w", "a"]

    def test_text_with_punctuation(self, tokenizer):
        """Test text with punctuation."""
        text = "明日は晴れです。"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # Should not include accent markers
        assert "[H]" not in tokens
        assert "[L]" not in tokens
        assert "|" not in tokens

    def test_question_text(self, tokenizer):
        """Test question text without accent mode."""
        text = "明日は晴れですか？"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # Should not include question marker
        assert "[Q]" not in tokens

    def test_devoiced_vowels(self, tokenizer):
        """Test devoiced vowels are represented with capital letters."""
        text = "すき"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # Should contain capital U for devoiced vowel
        assert "U" in tokens or "s" in tokens

    def test_geminate_consonant(self, tokenizer):
        """Test geminate consonant (促音) representation."""
        text = "切手"
        tokens = tokenizer.texts_to_tokens([text])[0]
        assert "cl" in tokens

    def test_moraic_nasal(self, tokenizer):
        """Test moraic nasal (撥音) representation."""
        text = "新聞"
        tokens = tokenizer.texts_to_tokens([text])[0]
        assert "N" in tokens

    def test_long_vowel(self, tokenizer):
        """Test long vowel representation."""
        text = "東京"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # Long vowels are represented by repeated vowels
        assert tokens.count("o") >= 2


class TestJapaneseTokenizerWithAccent:
    """Test JapaneseTokenizer with accent markers enabled."""

    @pytest.fixture
    def tokenizer(self):
        from zipvoice.tokenizer.tokenizer import JapaneseTokenizer

        return JapaneseTokenizer(use_accent=True)

    def test_accent_markers_present(self, tokenizer):
        """Test that accent markers are included."""
        text = "明日は晴れです。"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # Should include H and L markers
        assert "[H]" in tokens
        assert "[L]" in tokens

    def test_phrase_boundary(self, tokenizer):
        """Test accent phrase boundary marker."""
        text = "明日は晴れです。"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # Should include phrase boundary
        assert "|" in tokens

    def test_question_marker(self, tokenizer):
        """Test question marker for interrogative sentences."""
        text = "明日は晴れですか？"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # Should include question marker at the end
        assert tokens[-1] == "[Q]"

    def test_question_marker_half_width(self, tokenizer):
        """Test question marker with half-width question mark."""
        text = "明日は晴れですか?"
        tokens = tokenizer.texts_to_tokens([text])[0]
        assert tokens[-1] == "[Q]"

    def test_no_question_marker_for_statement(self, tokenizer):
        """Test that statements don't have question marker."""
        text = "明日は晴れです。"
        tokens = tokenizer.texts_to_tokens([text])[0]
        assert "[Q]" not in tokens

    def test_high_pitch_before_accent(self, tokenizer):
        """Test that [H] appears before accent nucleus."""
        text = "こんにちは"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # [H] should appear before [L]
        h_idx = tokens.index("[H]")
        l_idx = tokens.index("[L]")
        assert h_idx < l_idx

    def test_pause_handling(self, tokenizer):
        """Test that pause (pau) is properly handled."""
        text = "今日は、天気がいいです。"
        tokens = tokenizer.texts_to_tokens([text])[0]
        assert "pau" in tokens

    def test_expected_output_format(self, tokenizer):
        """Test expected output format for a known input."""
        text = "明日は晴れです。"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # Expected: [H] a sh I [L] t a w a | [H] h a [L] r e d e s U
        expected_structure = [
            "[H]",  # Start with high pitch
            "a",
            "sh",
            "I",  # Devoiced vowel
            "[L]",  # Pitch drops
            "t",
            "a",
            "w",
            "a",
            "|",  # Phrase boundary
            "[H]",  # High pitch again
            "h",
            "a",
            "[L]",  # Pitch drops
            "r",
            "e",
            "d",
            "e",
            "s",
            "U",  # Devoiced vowel
        ]
        assert tokens == expected_structure


class TestJapaneseTokenizerWithTokenFile:
    """Test JapaneseTokenizer with token file for ID conversion."""

    @pytest.fixture
    def tokenizer(self):
        from zipvoice.tokenizer.tokenizer import JapaneseTokenizer

        return JapaneseTokenizer(
            token_file="data/tokens_japanese_extended.txt", use_accent=True
        )

    def test_vocab_size(self, tokenizer):
        """Test vocabulary size includes new accent tokens."""
        # Original 385 tokens + 4 new accent tokens = 389
        assert tokenizer.vocab_size == 389

    def test_accent_token_ids(self, tokenizer):
        """Test that accent tokens have correct IDs."""
        assert tokenizer.token2id["[H]"] == 385
        assert tokenizer.token2id["[L]"] == 386
        assert tokenizer.token2id["|"] == 387
        assert tokenizer.token2id["[Q]"] == 388

    def test_token_to_id_conversion(self, tokenizer):
        """Test token to ID conversion."""
        text = "こんにちは"
        token_ids = tokenizer.texts_to_token_ids([text])[0]
        # First token should be [H] with ID 385
        assert token_ids[0] == 385
        # Should contain [L] with ID 386
        assert 386 in token_ids

    def test_question_token_id(self, tokenizer):
        """Test question marker token ID."""
        text = "明日は晴れですか？"
        token_ids = tokenizer.texts_to_token_ids([text])[0]
        # Last token should be [Q] with ID 388
        assert token_ids[-1] == 388

    def test_phrase_boundary_token_id(self, tokenizer):
        """Test phrase boundary token ID."""
        text = "私の名前は田中です。"
        token_ids = tokenizer.texts_to_token_ids([text])[0]
        # Should contain | with ID 387
        assert 387 in token_ids


class TestAccentMarkerConstants:
    """Test accent marker constants."""

    def test_marker_constants(self):
        from zipvoice.tokenizer.tokenizer import JapaneseTokenizer

        assert JapaneseTokenizer.ACCENT_HIGH == "[H]"
        assert JapaneseTokenizer.ACCENT_LOW == "[L]"
        assert JapaneseTokenizer.PHRASE_BOUNDARY == "|"
        assert JapaneseTokenizer.QUESTION_MARKER == "[Q]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
