"""The fast splitter must keep exact historical Jev payload boundaries."""
import random

import pytest

from mnemosyne.core import jev


def historical_chunks(text, size):
    if not text:
        return ['']
    result, part, length = [], [], 0
    for char in text:
        width = len(jev._json(char)) - 2
        if length + width > size and part:
            result.append(''.join(part))
            part, length = [], 0
        part.append(char)
        length += width
    if part:
        result.append(''.join(part))
    return result


@pytest.mark.parametrize('size', [1, 2, 3, 4, 7, 2000, 8000])
def test_exact_boundaries_for_json_escapes_and_unicode(size):
    text = ''.join(map(chr, range(128))) + 'éΩ中😀\u2028\u2029\U0010ffff'
    text *= 20
    assert jev.chunks(text, size) == historical_chunks(text, size)
    assert ''.join(jev.chunks(text, size)) == text


def test_deterministic_random_unicode_matches_original():
    rng = random.Random(20260921)
    for _ in range(40):
        codes = [rng.randrange(0x110000) for _ in range(500)]
        text = ''.join(chr(code) for code in codes if not 0xD800 <= code <= 0xDFFF)
        size = rng.randrange(1, 500)
        assert jev.chunks(text, size) == historical_chunks(text, size)


@pytest.mark.parametrize('text', ['\ud800', 'prefix\udfff'])
def test_invalid_surrogate_still_rejected(text):
    with pytest.raises(UnicodeEncodeError):
        jev.chunks(text)


def test_ascii_scan_does_not_serialize_each_character(monkeypatch):
    monkeypatch.setattr(jev, '_json', lambda *_: pytest.fail('Per-character serialization'))
    assert jev.chunks('a' * 10000, 2000) == ['a' * 2000] * 5
    assert jev.chunks('') == ['']
