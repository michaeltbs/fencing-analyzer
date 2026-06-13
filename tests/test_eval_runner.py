"""
test_eval_runner.py — Unit tests for eval_runner.py response parser.

Tests _parse_eval_response with single-line, multiline, and malformed inputs.
"""
import pytest

from eval_runner import _parse_eval_response


@pytest.mark.parametrize("text,expected_score,expected_issues,expected_suggestions", [
    (
        "SCORE: 4, ISSUES: a, b, SUGGESTIONS: c, d",
        4,
        ["a", "b"],
        ["c", "d"],
    ),
    (
        "SCORE: 3\nISSUES:\n- issue one\n- issue two\nSUGGESTIONS:\n- suggestion one",
        3,
        ["issue one", "issue two"],
        ["suggestion one"],
    ),
    (
        "Some intro text. SCORE: 5 ISSUES: none here SUGGESTIONS: keep going",
        5,
        ["none here"],
        ["keep going"],
    ),
    # Malformed: no structure -> fallback score 3 and whole text as suggestion
    (
        "I think this chunk is okay but could be better.",
        3,
        [],
        ["I think this chunk is okay but could be better."],
    ),
    # Score clamping
    (
        "SCORE: 10, ISSUES: x, SUGGESTIONS: y",
        5,
        ["x"],
        ["y"],
    ),
    # Score zero -> fallback 1
    (
        "SCORE: 0, ISSUES: bad, SUGGESTIONS: fix",
        1,
        ["bad"],
        ["fix"],
    ),
])
def test_parse_eval_response(text, expected_score, expected_issues, expected_suggestions):
    result = _parse_eval_response(text)
    assert result["score"] == expected_score
    assert result["issues"] == expected_issues
    assert result["suggestions"] == expected_suggestions
