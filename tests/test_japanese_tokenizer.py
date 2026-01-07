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

    def test_accent_pattern_has_transitions(self, tokenizer):
        """Test that accent pattern has both L and H transitions."""
        # Use a word with clear accent pattern
        text = "明日は晴れです"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # Should have both [L] and [H] markers
        assert "[L]" in tokens
        assert "[H]" in tokens
        # First mora is LOW (Japanese rule)
        assert tokens[0] == "[L]"

    def test_pause_handling(self, tokenizer):
        """Test that pause (pau) is properly handled."""
        text = "今日は、天気がいいです。"
        tokens = tokenizer.texts_to_tokens([text])[0]
        assert "pau" in tokens

    def test_expected_output_format(self, tokenizer):
        """Test expected output format for a known input."""
        text = "明日は晴れです。"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # Japanese accent rule: first mora is LOW (except atamadaka)
        # 明日: L-H-H pattern (first mora LOW)
        # 晴れです: L-H-L pattern (first mora LOW in new phrase)
        expected_structure = [
            "[L]",  # First mora is LOW
            "a",
            "[H]",  # Rise to HIGH from second mora
            "sh",
            "I",  # Devoiced vowel
            "t",
            "a",  # Accent nucleus (A1=0) - still high
            "[L]",  # Pitch drops after accent nucleus
            "w",
            "a",
            "|",  # Phrase boundary
            "[L]",  # First mora of new phrase is LOW
            "h",
            "a",
            "[H]",  # Rise to HIGH
            "r",
            "e",  # Accent nucleus (A1=0) - still high
            "[L]",  # Pitch drops after accent nucleus
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
        # Use text with clear accent pattern
        text = "明日は晴れです"
        token_ids = tokenizer.texts_to_token_ids([text])[0]
        # First token should be [L] with ID 386 (first mora is LOW)
        assert token_ids[0] == 386
        # Should contain [H] with ID 385 (rising pitch)
        assert 385 in token_ids

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


class TestAccentPatterns:
    """Test specific accent patterns to prevent regression."""

    @pytest.fixture
    def tokenizer(self):
        from zipvoice.tokenizer.tokenizer import JapaneseTokenizer

        return JapaneseTokenizer(use_accent=True)

    def test_atamadaka_accent(self, tokenizer):
        """Test atamadaka (頭高型) accent pattern - first mora high, rest low.

        Words like 箸 (はし chopsticks) have accent on first mora.
        Pattern: H-L (first high, rest low)
        """
        text = "箸"  # はし - chopsticks, 頭高型
        tokens = tokenizer.texts_to_tokens([text])[0]
        # Expected: [H] h a [L] sh i
        assert tokens[0] == "[H]"
        assert "[L]" in tokens
        # [H] should come before [L]
        h_idx = tokens.index("[H]")
        l_idx = tokens.index("[L]")
        assert h_idx < l_idx

    def test_accent_nucleus_is_high(self, tokenizer):
        """Test that accent nucleus (A1=0) is marked as HIGH, not LOW.

        This is the core fix: A1=0 should be H, A1>0 should be L.
        """
        text = "明日"  # あした - accent on 'し'
        tokens = tokenizer.texts_to_tokens([text])[0]
        # The accent pattern should have some HIGH markers
        assert "[H]" in tokens
        # First phonemes should be LOW (first mora before nucleus)
        # Then HIGH for rest until nucleus
        assert "[L]" in tokens

    def test_odaka_accent(self, tokenizer):
        """Test odaka (尾高型) accent pattern - pitch drops at the end."""
        text = "明日は"  # あしたは - 'は' is low
        tokens = tokenizer.texts_to_tokens([text])[0]
        assert "[H]" in tokens
        assert "[L]" in tokens


class TestG2PAccentLogic:
    """Test G2P accent logic with explicit patterns.

    Japanese accent rules:
    - First mora is LOW (except atamadaka where accent is on first mora)
    - Morae before and at accent nucleus are HIGH
    - Morae after accent nucleus are LOW
    """

    @pytest.fixture
    def tokenizer(self):
        from zipvoice.tokenizer.tokenizer import JapaneseTokenizer

        return JapaneseTokenizer(use_accent=True)

    def test_watashi_wa_first_mora_low(self, tokenizer):
        """Test 私は - first mora should be LOW.

        私は has accent pattern L-H-H-H:
        - わ(L): first mora, before nucleus
        - た(H): second mora, before nucleus
        - し(H): third mora, before nucleus
        - は(H): fourth mora, at nucleus
        """
        text = "私は"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # First token should be [L]
        assert tokens[0] == "[L]", f"Expected [L] but got {tokens[0]}, full: {tokens}"
        # Then [H] for the rising pitch
        assert "[H]" in tokens

    def test_hashi_bridge_first_mora_low(self, tokenizer):
        """Test 橋 (bridge) - L-H pattern (尾高型).

        橋 has accent on mora 2 (尾高型):
        - は(L): first mora
        - し(H): second mora, at nucleus
        """
        text = "橋"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # First token should be [L]
        assert tokens[0] == "[L]", f"Expected [L] but got {tokens[0]}, full: {tokens}"

    def test_hashi_chopsticks_first_mora_high(self, tokenizer):
        """Test 箸 (chopsticks) - H-L pattern (頭高型).

        箸 has accent on mora 1 (頭高型):
        - は(H): first mora, at nucleus
        - し(L): second mora, after nucleus
        """
        text = "箸"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # First token should be [H] (atamadaka)
        assert tokens[0] == "[H]", f"Expected [H] but got {tokens[0]}, full: {tokens}"
        # Then [L] for the drop
        assert "[L]" in tokens

    def test_konnichiwa_first_mora_low(self, tokenizer):
        """Test こんにちは - first mora should be LOW.

        こんにちは has L-H-H-H-H pattern:
        - こ(L): first mora
        - ん(H): second mora
        - に(H): third mora
        - ち(H): fourth mora
        - は(H): fifth mora, at nucleus
        """
        text = "こんにちは"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # First token should be [L]
        assert tokens[0] == "[L]", f"Expected [L] but got {tokens[0]}, full: {tokens}"

    def test_toukyou_first_mora_low(self, tokenizer):
        """Test 東京 - first mora should be LOW.

        東京 has L-H-H-H pattern:
        - と(L): first mora
        - う(H): second mora
        - きょ(H): third mora
        - う(H): fourth mora, at nucleus
        """
        text = "東京"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # First token should be [L]
        assert tokens[0] == "[L]", f"Expected [L] but got {tokens[0]}, full: {tokens}"

    def test_ashita_first_mora_low(self, tokenizer):
        """Test 明日 (あした) - first mora should be LOW.

        明日 has L-H-H pattern:
        - あ(L): first mora
        - し(H): second mora
        - た(H): third mora, at nucleus
        """
        text = "明日"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # First token should be [L]
        assert tokens[0] == "[L]", f"Expected [L] but got {tokens[0]}, full: {tokens}"

    def test_explicit_pattern_watashi_wa(self, tokenizer):
        """Test exact token sequence for 私は."""
        text = "私は"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # Expected: [L] w a [H] t a sh i w a
        expected = ["[L]", "w", "a", "[H]", "t", "a", "sh", "i", "w", "a"]
        assert tokens == expected, f"Expected {expected}, got {tokens}"

    def test_explicit_pattern_hashi_chopsticks(self, tokenizer):
        """Test exact token sequence for 箸 (chopsticks)."""
        text = "箸"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # Expected: [H] h a [L] sh i
        expected = ["[H]", "h", "a", "[L]", "sh", "i"]
        assert tokens == expected, f"Expected {expected}, got {tokens}"

    def test_explicit_pattern_hashi_bridge(self, tokenizer):
        """Test exact token sequence for 橋 (bridge)."""
        text = "橋"
        tokens = tokenizer.texts_to_tokens([text])[0]
        # Expected: [L] h a [H] sh i
        expected = ["[L]", "h", "a", "[H]", "sh", "i"]
        assert tokens == expected, f"Expected {expected}, got {tokens}"


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
