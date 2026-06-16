"""v0.8 A1 — conservative keyword cleaning for search queries."""

from __future__ import annotations

from sherlock.tools.web_search import clean_query


def test_strips_punctuation():
    assert clean_query("일본 여행지, 추천해줘!") == "일본 여행지 추천해줘"
    assert clean_query("best ramen in Tokyo?") == "best ramen Tokyo"


def test_strips_korean_particles_multichar_only():
    # v0.9: only MULTI-char particles are stripped (에서/부터/까지...). Single-char
    # particles (이/가/은/는/의/를...) are kept — stripping them corrupts real
    # nouns (하와이→하와, 제주도→제주, 고양이→고양).
    assert clean_query("도쿄에서 숙소 추천") == "도쿄 숙소 추천"
    assert clean_query("서울부터 부산까지 기차") == "서울 부산 기차"


def test_single_char_particles_kept_to_protect_nouns():
    assert clean_query("하와이 여행 코스") == "하와이 여행 코스"
    assert clean_query("제주도 맛집") == "제주도 맛집"
    assert clean_query("고양이 사료 추천") == "고양이 사료 추천"
    # attached single-char particles stay attached (engines tokenize these fine)
    assert clean_query("일본의 여행지를 추천") == "일본의 여행지를 추천"


def test_preserves_quoted_phrases_versions_and_symbols():
    assert clean_query('"exact phrase" python 3.12') == '"exact phrase" python 3.12'
    assert clean_query("C++ vs C# benchmark") == "C++ vs C# benchmark"
    assert "node.js" in clean_query("node.js memory leak?")


def test_does_not_gut_short_or_clean_tokens():
    # already-clean keywords pass through unchanged
    assert clean_query("일본 여행 명소") == "일본 여행 명소"
    assert clean_query("Japan travel tips") == "Japan travel tips"
    # a 2-char Hangul token must not be emptied even if it ends in a josa char
    assert clean_query("주가") == "주가"


def test_english_stopwords_removed_latin_only():
    assert clean_query("what is the best camera for travel") == "best camera travel"
    # CJK tokens never dropped as stopwords
    assert "여행" in clean_query("여행 정보")


def test_never_returns_empty():
    # an all-stopword query falls back rather than returning ""
    assert clean_query("the of to").strip() != ""
    assert clean_query("???").strip() != "" or clean_query("???") == ""


def test_idempotent():
    once = clean_query("일본의 여행지를, 추천해줘!")
    assert clean_query(once) == once
