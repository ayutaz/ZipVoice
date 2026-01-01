# Copyright      2023-2024  Xiaomi Corp.        (authors: Zengwei Yao
#                                                         Han Zhu,
#                                                         Wei Kang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
from abc import ABC, abstractmethod
from functools import lru_cache, reduce
from typing import Dict, List, Optional

import jieba
from lhotse import CutSet
from pypinyin import Style, lazy_pinyin
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

from zipvoice.tokenizer.normalizer import (
    ChineseTextNormalizer,
    EnglishTextNormalizer,
    JapaneseTextNormalizer,
)

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f \
            https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

try:
    import pyopenjtalk
except ImportError:
    pyopenjtalk = None

jieba.default_logger.setLevel(logging.INFO)


# Cached lazy_pinyin for performance optimization
@lru_cache(maxsize=10000)
def _cached_lazy_pinyin_word(word: str) -> tuple:
    """Cache pinyin conversion for individual words."""
    return tuple(
        lazy_pinyin(
            [word],
            style=Style.TONE3,
            tone_sandhi=True,
            neutral_tone_with_five=True,
        )
    )


def cached_lazy_pinyin(segs: list) -> list:
    """Convert a list of words to pinyin using cached results."""
    result = []
    for word in segs:
        result.extend(_cached_lazy_pinyin_word(word))
    return result


class Tokenizer(ABC):
    """Abstract base class for tokenizers, defining common interface."""

    @abstractmethod
    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert list of texts to list of token id sequences."""
        raise NotImplementedError

    @abstractmethod
    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        """Convert list of texts to list of token sequences."""
        raise NotImplementedError

    @abstractmethod
    def tokens_to_token_ids(self, tokens: List[List[str]]) -> List[List[int]]:
        """Convert list of token sequences to list of token id sequences."""
        raise NotImplementedError


class SimpleTokenizer(Tokenizer):
    """The simplpest tokenizer, treat every character as a token,
    without text normalization.
    """

    def __init__(self, token_file: Optional[str] = None):
        """
        Args:
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line.
        """
        # Parse token file
        self.has_tokens = False
        if token_file is None:
            logging.debug(
                "Initialize Tokenizer without tokens file, \
                will fail when map to ids."
            )
            return
        self.token2id: Dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id
        self.pad_id = self.token2id["_"]  # padding
        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(
        self,
        texts: List[str],
    ) -> List[List[str]]:
        tokens_list = [list(texts[i]) for i in range(len(texts))]
        return tokens_list

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."

        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.debug(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list


class EspeakTokenizer(Tokenizer):
    """A simple tokenizer with Espeak g2p function."""

    def __init__(self, token_file: Optional[str] = None, lang: str = "en-us"):
        """
        Args:
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line.
          lang: the language identifier, see
            https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md
        """
        # Parse token file
        self.has_tokens = False
        self.lang = lang
        if token_file is None:
            logging.debug(
                "Initialize Tokenizer without tokens file, \
                will fail when map to ids."
            )
            return
        self.token2id: Dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id
        self.pad_id = self.token2id["_"]  # padding
        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def g2p(self, text: str) -> List[str]:
        try:
            tokens = phonemize_espeak(text, self.lang)
            tokens = reduce(lambda x, y: x + y, tokens)
            return tokens
        except Exception as ex:
            logging.warning(f"Tokenization of {self.lang} texts failed: {ex}")
            return []

    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(
        self,
        texts: List[str],
    ) -> List[List[str]]:
        tokens_list = [self.g2p(texts[i]) for i in range(len(texts))]
        return tokens_list

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."

        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.debug(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list


class EmiliaTokenizer(Tokenizer):
    # Pre-compiled regex patterns for performance
    _PART_PATTERN = re.compile(r"[<[].*?[>\]]|.")
    _SPLIT_PATTERN = re.compile(r"([<[].*?[>\]])")

    def __init__(self, token_file: Optional[str] = None, token_type="phone"):
        """
        Args:
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line.
        """
        assert (
            token_type == "phone"
        ), f"Only support phone tokenizer for Emilia, but get {token_type}."

        self.english_normalizer = EnglishTextNormalizer()
        self.chinese_normalizer = ChineseTextNormalizer()

        self.has_tokens = False
        if token_file is None:
            logging.debug(
                "Initialize Tokenizer without tokens file, \
                    will fail when map to ids."
            )
            return
        self.token2id: Dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id
        self.pad_id = self.token2id["_"]  # padding

        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def preprocess_text(
        self,
        text: str,
    ) -> str:
        return self.map_punctuations(text)

    def texts_to_tokens(
        self,
        texts: List[str],
    ) -> List[List[str]]:
        for i in range(len(texts)):
            # Text normalization
            texts[i] = self.preprocess_text(texts[i])

        phoneme_list = []
        for text in texts:
            # now only en and ch
            segments = self.get_segment(text)
            all_phoneme = []
            for index in range(len(segments)):
                seg = segments[index]
                if seg[1] == "zh":
                    phoneme = self.tokenize_ZH(seg[0])
                elif seg[1] == "en":
                    phoneme = self.tokenize_EN(seg[0])
                elif seg[1] == "pinyin":
                    phoneme = self.tokenize_pinyin(seg[0])
                elif seg[1] == "tag":
                    phoneme = [seg[0]]
                else:
                    logging.warning(
                        f"No English or Chinese characters found, \
                            skipping segment of unknown language: {seg}"
                    )
                    continue
                all_phoneme += phoneme
            phoneme_list.append(all_phoneme)
        return phoneme_list

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."
        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.debug(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list

    def tokenize_ZH(self, text: str) -> List[str]:
        try:
            text = self.chinese_normalizer.normalize(text)
            segs = list(jieba.cut(text))
            full = cached_lazy_pinyin(segs)
            phones = []
            for x in full:
                # valid pinyin (in tone3 style) is alphabet + 1 number in [1-5].
                if not (x[0:-1].isalpha() and x[-1] in ("1", "2", "3", "4", "5")):
                    phones.append(x)
                    continue
                else:
                    phones.extend(self.seperate_pinyin(x))
            return phones
        except Exception as ex:
            logging.warning(f"Tokenization of Chinese texts failed: {ex}")
            return []

    def tokenize_EN(self, text: str) -> List[str]:
        try:
            text = self.english_normalizer.normalize(text)
            tokens = phonemize_espeak(text, "en-us")
            tokens = reduce(lambda x, y: x + y, tokens)
            return tokens
        except Exception as ex:
            logging.warning(f"Tokenization of English texts failed: {ex}")
            return []

    def tokenize_pinyin(self, text: str) -> List[str]:
        try:
            assert text.startswith("<") and text.endswith(">")
            text = text.lstrip("<").rstrip(">")
            # valid pinyin (in tone3 style) is alphabet + 1 number in [1-5].
            if not (text[0:-1].isalpha() and text[-1] in ("1", "2", "3", "4", "5")):
                logging.warning(
                    f"Strings enclosed with <> should be pinyin, \
                    but got: {text}. Skipped it. "
                )
                return []
            else:
                return self.seperate_pinyin(text)
        except Exception as ex:
            logging.warning(f"Tokenize pinyin failed: {ex}")
            return []

    def seperate_pinyin(self, text: str) -> List[str]:
        """
        Separate pinyin into initial and final
        """
        pinyins = []
        initial = to_initials(text, strict=False)
        # don't want to share tokens with espeak tokens,
        # so use tone3 style
        final = to_finals_tone3(
            text,
            strict=False,
            neutral_tone_with_five=True,
        )
        if initial != "":
            # don't want to share tokens with espeak tokens,
            # so add a '0' after each initial
            pinyins.append(initial + "0")
        if final != "":
            pinyins.append(final)
        return pinyins

    def map_punctuations(self, text):
        text = text.replace("，", ",")
        text = text.replace("。", ".")
        text = text.replace("！", "!")
        text = text.replace("？", "?")
        text = text.replace("；", ";")
        text = text.replace("：", ":")
        text = text.replace("、", ",")
        text = text.replace("‘", "'")
        text = text.replace("“", '"')
        text = text.replace("”", '"')
        text = text.replace("’", "'")
        text = text.replace("⋯", "…")
        text = text.replace("···", "…")
        text = text.replace("・・・", "…")
        text = text.replace("...", "…")
        return text

    def get_segment(self, text: str) -> List[str]:
        """
        Split a text into segments based on language types
        (Chinese, English, Pinyin, tags, etc.)

        Args:
            text (str): Input text to be segmented

        Returns:
            List[str]: Segmented text parts with their language types

        Example:
            Input: 我们是小米人,是吗? Yes I think so!霍...啦啦啦
            Output: [('我们是小米人,是吗? ', 'zh'),
                ('Yes I think so!', 'en'), ('霍...啦啦啦', 'zh')]
        """
        # Stores the final segmented parts and their language types
        segments = []
        # Stores the language type of each character in the input text
        types = []
        temp_seg = ""
        temp_lang = ""

        # Each part is a character, or a special string enclosed in <> and []
        # <> denotes pinyin string, [] denotes other special strings.
        text = self._PART_PATTERN.findall(text)

        for i, part in enumerate(text):
            if self.is_chinese(part) or self.is_pinyin(part):
                types.append("zh")
            elif self.is_alphabet(part):
                types.append("en")
            else:
                types.append("other")

        assert len(types) == len(text)

        for i in range(len(types)):
            # find the first char of the seg
            if i == 0:
                temp_seg += text[i]
                temp_lang = types[i]
            else:
                if temp_lang == "other":
                    temp_seg += text[i]
                    temp_lang = types[i]
                else:
                    if types[i] in [temp_lang, "other"]:
                        temp_seg += text[i]
                    else:
                        segments.append((temp_seg, temp_lang))
                        temp_seg = text[i]
                        temp_lang = types[i]

        segments.append((temp_seg, temp_lang))

        # Handle "pinyin" and "tag" types
        segments = self.split_segments(segments)
        return segments

    def split_segments(self, segments):
        """
        split segments into smaller parts if special strings enclosed by [] or <>
        are found, where <> denotes pinyin strings, [] denotes other special strings.

        Args:
            segments (list): A list of tuples where each tuple contains:
                - temp_seg (str): The text segment to be split.
                - temp_lang (str): The language code associated with the segment.

        Returns:
            list: A list of smaller segments.
        """
        result = []
        for temp_seg, temp_lang in segments:
            parts = self._SPLIT_PATTERN.split(temp_seg)
            for part in parts:
                if not part:
                    continue
                if self.is_pinyin(part):
                    result.append((part, "pinyin"))
                elif self.is_tag(part):
                    result.append((part, "tag"))
                else:
                    result.append((part, temp_lang))
        return result

    def is_chinese(self, char: str) -> bool:
        if char >= "\u4e00" and char <= "\u9fa5":
            return True
        else:
            return False

    def is_alphabet(self, char: str) -> bool:
        if (char >= "\u0041" and char <= "\u005a") or (
            char >= "\u0061" and char <= "\u007a"
        ):
            return True
        else:
            return False

    def is_pinyin(self, part: str) -> bool:
        if part.startswith("<") and part.endswith(">"):
            return True
        else:
            return False

    def is_tag(self, part: str) -> bool:
        if part.startswith("[") and part.endswith("]"):
            return True
        else:
            return False


class DialogTokenizer(EmiliaTokenizer):
    def __init__(self, token_file: Optional[str] = None, token_type="phone"):
        super().__init__(token_file=token_file, token_type=token_type)
        if token_file:
            self.spk_a_id = self.token2id["[S1]"]
            self.spk_b_id = self.token2id["[S2]"]

    def preprocess_text(
        self,
        text: str,
    ) -> str:
        text = re.sub(r"\s*(\[S[12]\])\s*", r"\1", text)
        text = self.map_punctuations(text)
        return text


class JapaneseTokenizer(Tokenizer):
    """Tokenizer for Japanese text using pyopenjtalk for G2P.

    Supports Japanese-English mixed text by automatically detecting
    and processing each language segment appropriately.

    Args:
        token_file: Path to the token file with '{token}\\t{token_id}' per line.
        use_accent: If True, include accent markers ([H], [L], |, [Q]) in output.
    """

    # Pre-compiled regex patterns for performance
    _PART_PATTERN = re.compile(r"[<[].*?[>\]]|.")
    _SPLIT_PATTERN = re.compile(r"([<[].*?[>\]])")

    # Accent markers
    ACCENT_HIGH = "[H]"  # High pitch region (before accent nucleus)
    ACCENT_LOW = "[L]"   # Low pitch region (at/after accent nucleus)
    PHRASE_BOUNDARY = "|"  # Accent phrase boundary
    QUESTION_MARKER = "[Q]"  # Question intonation marker

    def __init__(self, token_file: Optional[str] = None, use_accent: bool = True):
        """
        Args:
            token_file: Path to the token file with '{token}\\t{token_id}' per line.
            use_accent: If True, include accent markers in output.
        """
        if pyopenjtalk is None:
            raise RuntimeError(
                "pyopenjtalk is not installed. Please run:\n"
                "pip install pyopenjtalk-plus"
            )

        self.use_accent = use_accent
        self.japanese_normalizer = JapaneseTextNormalizer()
        self.english_normalizer = EnglishTextNormalizer()

        self.has_tokens = False
        if token_file is None:
            logging.debug(
                "Initialize Tokenizer without tokens file, "
                "will fail when map to ids."
            )
            return

        self.token2id: Dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id

        self.pad_id = self.token2id["_"]  # padding
        # vocab_size should be max_id + 1 to handle sparse IDs
        self.vocab_size = max(self.token2id.values()) + 1
        self.has_tokens = True

    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def preprocess_text(self, text: str) -> str:
        """Preprocess text: normalize punctuation."""
        return self.map_punctuations(text)

    def texts_to_tokens(
        self,
        texts: List[str],
    ) -> List[List[str]]:
        for i in range(len(texts)):
            texts[i] = self.preprocess_text(texts[i])

        phoneme_list = []
        for text in texts:
            segments = self.get_segment(text)
            all_phoneme = []
            for seg in segments:
                if seg[1] == "ja":
                    phoneme = self.tokenize_JA(seg[0])
                elif seg[1] == "en":
                    phoneme = self.tokenize_EN(seg[0])
                elif seg[1] == "tag":
                    phoneme = [seg[0]]
                else:
                    logging.warning(
                        f"No Japanese or English characters found, "
                        f"skipping segment of unknown language: {seg}"
                    )
                    continue
                all_phoneme += phoneme
            phoneme_list.append(all_phoneme)
        return phoneme_list

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."
        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.debug(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list

    def tokenize_JA(self, text: str) -> List[str]:
        """Convert Japanese text to phonemes using pyopenjtalk.

        Args:
            text: Japanese text to convert.

        Returns:
            List of phoneme tokens. If use_accent is True, includes accent markers.
        """
        try:
            original_text = text
            text = self.japanese_normalizer.normalize(text)

            if self.use_accent:
                return self._tokenize_JA_with_accent(text, original_text)
            else:
                # pyopenjtalk.g2p returns space-separated phonemes
                phonemes_str = pyopenjtalk.g2p(text)
                # Split by space to get individual phonemes
                phonemes = phonemes_str.split()
                return phonemes
        except Exception as ex:
            logging.warning(f"Tokenization of Japanese texts failed: {ex}")
            return []

    def _tokenize_JA_with_accent(self, text: str, original_text: str) -> List[str]:
        """Convert Japanese text to phonemes with accent markers.

        Uses pyopenjtalk full context labels to extract:
        - [H]: High pitch region (before accent nucleus)
        - [L]: Low pitch region (at/after accent nucleus)
        - |: Accent phrase boundary
        - [Q]: Question marker (for sentences ending with ?)

        Args:
            text: Normalized Japanese text.
            original_text: Original text (for question detection).

        Returns:
            List of phoneme tokens with accent markers.
        """
        labels = pyopenjtalk.extract_fullcontext(text)

        result = []
        prev_level = None
        prev_f = None

        for label in labels:
            parts = label.split("/")
            phone_part = parts[0]
            phone = phone_part.split("-")[1].split("+")[0]

            # Skip silence markers
            if phone == "sil":
                continue

            # Handle pause
            if phone == "pau":
                result.append("pau")
                prev_f = None
                prev_level = None
                continue

            # Get accent phrase info (F field)
            f_field = next((p for p in parts if p.startswith("F:")), "F:xx_xx")
            f_values = f_field[2:].split("#")[0]

            # Accent phrase boundary
            if prev_f is not None and f_values != prev_f:
                result.append(self.PHRASE_BOUNDARY)
                prev_level = None

            # Get accent info (A field)
            a_field = next((p for p in parts if p.startswith("A:")), "A:xx+xx+xx")
            a1 = a_field.split("+")[0][2:]

            # Determine pitch level
            try:
                a1_val = int(a1)
                level = "H" if a1_val < 0 else "L"
            except ValueError:
                level = "L"  # Default to low for unknown

            # Add pitch marker if level changed
            if prev_level != level:
                result.append(self.ACCENT_HIGH if level == "H" else self.ACCENT_LOW)

            result.append(phone)
            prev_level = level
            prev_f = f_values

        # Add question marker if text ends with ?
        if original_text.rstrip().endswith("?") or original_text.rstrip().endswith("？"):
            result.append(self.QUESTION_MARKER)

        return result

    def tokenize_EN(self, text: str) -> List[str]:
        """Convert English text to phonemes using espeak.

        Args:
            text: English text to convert.

        Returns:
            List of phoneme tokens.
        """
        try:
            text = self.english_normalizer.normalize(text)
            tokens = phonemize_espeak(text, "en-us")
            tokens = reduce(lambda x, y: x + y, tokens)
            return tokens
        except Exception as ex:
            logging.warning(f"Tokenization of English texts failed: {ex}")
            return []

    def map_punctuations(self, text: str) -> str:
        """Map Japanese punctuation to ASCII equivalents."""
        text = text.replace("，", ",")
        text = text.replace("。", ".")
        text = text.replace("！", "!")
        text = text.replace("？", "?")
        text = text.replace("；", ";")
        text = text.replace("：", ":")
        text = text.replace("、", ",")
        text = text.replace("'", "'")
        text = text.replace(""", '"')
        text = text.replace(""", '"')
        text = text.replace("'", "'")
        text = text.replace("⋯", "…")
        text = text.replace("···", "…")
        text = text.replace("・・・", "…")
        text = text.replace("...", "…")
        text = text.replace("「", '"')
        text = text.replace("」", '"')
        text = text.replace("『", '"')
        text = text.replace("』", '"')
        return text

    def get_segment(self, text: str) -> List[tuple]:
        """Split text into segments based on language (Japanese/English).

        Args:
            text: Input text to be segmented.

        Returns:
            List of tuples (segment_text, language_type).
            Language types: 'ja', 'en', 'tag', 'other'
        """
        segments = []
        types = []
        temp_seg = ""
        temp_lang = ""

        # Split text into characters or special markers
        text_parts = self._PART_PATTERN.findall(text)

        for part in text_parts:
            if self.is_japanese(part):
                types.append("ja")
            elif self.is_alphabet(part):
                types.append("en")
            else:
                types.append("other")

        assert len(types) == len(text_parts)

        for i in range(len(types)):
            if i == 0:
                temp_seg += text_parts[i]
                temp_lang = types[i]
            else:
                if temp_lang == "other":
                    temp_seg += text_parts[i]
                    temp_lang = types[i]
                else:
                    if types[i] in [temp_lang, "other"]:
                        temp_seg += text_parts[i]
                    else:
                        segments.append((temp_seg, temp_lang))
                        temp_seg = text_parts[i]
                        temp_lang = types[i]

        segments.append((temp_seg, temp_lang))

        # Handle tags
        segments = self.split_segments(segments)
        return segments

    def split_segments(self, segments: List[tuple]) -> List[tuple]:
        """Split segments to handle special tags enclosed in []."""
        result = []
        for temp_seg, temp_lang in segments:
            parts = self._SPLIT_PATTERN.split(temp_seg)
            for part in parts:
                if not part:
                    continue
                if self.is_tag(part):
                    result.append((part, "tag"))
                else:
                    result.append((part, temp_lang))
        return result

    def is_japanese(self, char: str) -> bool:
        """Check if a character is Japanese (hiragana, katakana, or kanji)."""
        # Hiragana: U+3040-U+309F
        if "\u3040" <= char <= "\u309f":
            return True
        # Katakana: U+30A0-U+30FF
        if "\u30a0" <= char <= "\u30ff":
            return True
        # CJK Unified Ideographs (Kanji): U+4E00-U+9FAF
        if "\u4e00" <= char <= "\u9faf":
            return True
        # Katakana Phonetic Extensions: U+31F0-U+31FF
        if "\u31f0" <= char <= "\u31ff":
            return True
        # Halfwidth Katakana: U+FF65-U+FF9F
        if "\uff65" <= char <= "\uff9f":
            return True
        return False

    def is_alphabet(self, char: str) -> bool:
        """Check if a character is an ASCII alphabet letter."""
        if ("\u0041" <= char <= "\u005a") or ("\u0061" <= char <= "\u007a"):
            return True
        return False

    def is_tag(self, part: str) -> bool:
        """Check if a part is a special tag enclosed in []."""
        if part.startswith("[") and part.endswith("]"):
            return True
        return False


class LibriTTSTokenizer(Tokenizer):
    def __init__(self, token_file: Optional[str] = None, token_type="char"):
        """
        Args:
          type: the type of tokenizer, e.g., bpe, char, phone.
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line if type is
            char or phone, otherwise it is a bpe_model file.
        """
        self.type = token_type
        assert token_type in ["bpe", "char", "phone"]
        try:
            import tacotron_cleaner.cleaners
        except Exception as ex:
            raise RuntimeError(f"{ex}\nPlease run\n" "pip install espnet_tts_frontend")

        self.normalize = tacotron_cleaner.cleaners.custom_english_cleaners

        self.has_tokens = False
        if token_file is None:
            logging.debug(
                "Initialize Tokenizer without tokens file, \
                will fail when map to ids."
            )
            return
        if token_type == "bpe":
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor()
            self.sp.load(token_file)
            self.pad_id = self.sp.piece_to_id("<pad>")
            self.vocab_size = self.sp.get_piece_size()
        else:
            self.token2id: Dict[str, int] = {}
            with open(token_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    info = line.rstrip().split("\t")
                    token, id = info[0], int(info[1])
                    assert token not in self.token2id, token
                    self.token2id[token] = id
            self.pad_id = self.token2id["_"]  # padding
            self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        if self.type == "bpe":
            for i in range(len(texts)):
                texts[i] = self.normalize(texts[i])
            return self.sp.encode(texts)
        else:
            return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(
        self,
        texts: List[str],
    ) -> List[List[str]]:
        for i in range(len(texts)):
            texts[i] = self.normalize(texts[i])

        if self.type == "char":
            tokens_list = [list(texts[i]) for i in range(len(texts))]
        elif self.type == "phone":
            tokens_list = [
                phonemize_espeak(texts[i].lower(), "en-us") for i in range(len(texts))
            ]
        elif self.type == "bpe":
            tokens_list = self.sp.encode(texts, out_type=str)

        return tokens_list

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."

        assert self.type != "bpe", "BPE tokenizer does not support this function."

        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.debug(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list


def add_tokens(cut_set: CutSet, tokenizer: str, lang: str):
    if tokenizer == "emilia":
        tokenizer = EmiliaTokenizer()
    elif tokenizer == "espeak":
        tokenizer = EspeakTokenizer(lang=lang)
    elif tokenizer == "dialog":
        tokenizer = DialogTokenizer()
    elif tokenizer == "libritts":
        tokenizer = LibriTTSTokenizer()
    elif tokenizer == "simple":
        tokenizer = SimpleTokenizer()
    elif tokenizer == "japanese":
        tokenizer = JapaneseTokenizer()
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer}.")

    def _prepare_cut(cut):
        # Each cut only contains one supervision
        assert len(cut.supervisions) == 1, (len(cut.supervisions), cut)
        text = cut.supervisions[0].text
        tokens = tokenizer.texts_to_tokens([text])[0]
        cut.supervisions[0].tokens = tokens
        return cut

    cut_set = cut_set.map(_prepare_cut)
    return cut_set


if __name__ == "__main__":
    text = (
        "我们是5年小米人,是吗? Yes I think so! "
        "mr king, 5 years, from 2019 to 2024."
        "霍...啦啦啦超过90%的人<le5>...?!9204"
    )
    tokenizer = EmiliaTokenizer()
    tokens = tokenizer.texts_to_tokens([text])
    print(f"tokens: {'|'.join(tokens[0])}")
