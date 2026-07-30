# SPDX-License-Identifier: MIT
"""
Tests for the Prompt Translator.

The oracle for the dictionary is the published table itself: the tests parse
Table 3 out of paper/supplementary.tex rather than restating it, so the
released module and the paper cannot drift apart silently.
"""

import os
import re

import pytest

from neuromorphic_prompt_translator import (
    CognitiveFunction,
    EmotionalArc,
    GrammarRules,
    NeuromorphicTranslator,
    NeuromorphicVocabulary,
    SalienceAnalyzer,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPLEMENTARY_TEX = os.path.join(REPO_ROOT, "paper", "supplementary.tex")

# Body-part nouns that must never be left dangling next to an emotional phrase
BODY_PARTS = ("head", "eyes", "eye", "jaw", "shoulders", "hand", "hands", "mouth")


def published_dictionary():
    """Parse the 22-entry motion-to-emotion table from the supplementary."""
    tex = open(SUPPLEMENTARY_TEX, encoding="utf-8").read()
    table = tex[tex.index("tab:pt_dictionary"):]
    body = re.search(r"\\midrule(.*?)\\bottomrule", table, re.S).group(1)

    entries = {}
    for line in body.strip().splitlines():
        if "&" not in line:
            continue
        literal, emotional = line.split("&", 1)
        entries[literal.strip()] = emotional.replace("\\\\", "").strip()
    return entries


@pytest.fixture(scope="module")
def translator():
    return NeuromorphicTranslator()


# ---------------------------------------------------------------------------
# Published dictionary parity
# ---------------------------------------------------------------------------

def test_supplementary_table_parses_to_22_entries():
    assert len(published_dictionary()) == 22


def test_motion_dictionary_matches_the_published_table():
    """Every published mapping is implemented, verbatim, with no extras."""
    assert NeuromorphicVocabulary.MOTION_TO_EMOTION == published_dictionary()


@pytest.mark.parametrize("literal,emotional", sorted(published_dictionary().items()))
def test_each_published_mapping_is_applied(translator, literal, emotional):
    out = translator.translate(literal, preserve_identity=False)
    assert emotional in out.lower() or emotional.capitalize() in out


def test_extensions_never_shadow_a_published_mapping():
    published = NeuromorphicVocabulary.MOTION_TO_EMOTION
    merged = NeuromorphicVocabulary.all_mappings()
    for key, value in published.items():
        assert merged[key] == value


def test_extensions_are_not_silently_part_of_the_published_table():
    extended = NeuromorphicVocabulary.EXTENDED_MOTION_TO_EMOTION
    assert set(extended) - set(published_dictionary()) == set(extended)


# ---------------------------------------------------------------------------
# Substitution invariants
# ---------------------------------------------------------------------------

def test_no_replacement_text_contains_a_dictionary_key(translator):
    """Order independence: a substituted phrase can never be re-substituted."""
    mappings = NeuromorphicVocabulary.all_mappings()
    offenders = []
    for key, value in mappings.items():
        for other in mappings:
            if re.search(rf"\b{re.escape(other)}\b", value, flags=re.IGNORECASE):
                offenders.append((key, other))
    assert offenders == []


def test_longest_key_wins_over_its_prefix(translator):
    """'moves closer' is published; the bare 'moves' extension must not win."""
    out = translator.translate("she moves closer", preserve_identity=False)
    assert "warmth drawing near" in out
    assert "shifting with purposeful energy" not in out


def test_leans_forward_and_back_are_distinct(translator):
    forward = translator.translate("he leans forward", preserve_identity=False)
    back = translator.translate("he leans back", preserve_identity=False)
    assert "engagement intensifying" in forward
    assert "contemplative withdrawal" in back


@pytest.mark.parametrize("prompt", [
    "woman moves head",
    "man nods head",
    "she blinks her eyes",
    "Victorian woman portrait, subtle head movement, slight smile, blinking eyes",
    "man gestures while talking, nods head, looks around the room",
    "person turns to face camera, moves closer, speaks",
])
def test_no_stranded_body_part_after_translation(translator, prompt):
    """A verb + object phrase must translate as a unit, not leave the object."""
    out = translator.translate(prompt, preserve_identity=False).lower()
    for noun in BODY_PARTS:
        assert not re.search(rf"(attention|processing|settling|energy|agreement)\s+{noun}\b", out), out


def test_turns_to_face_reads_as_a_direction(translator):
    out = translator.translate("person turns to face camera", preserve_identity=False)
    assert "attention shifting with purpose toward camera" in out
    assert "to face" not in out


def test_moves_head_folds_onto_head_movement(translator):
    out = translator.translate("woman moves head", preserve_identity=False)
    assert "subtle shift in attention" in out
    assert "head" not in out.lower()


def test_duplicate_adjective_is_collapsed(translator):
    out = translator.translate("subtle head movement", preserve_identity=False)
    assert out.lower().count("subtle") == 1


def test_unmapped_text_is_left_alone(translator):
    out = translator.translate("Victorian study, gaslight",
                               preserve_identity=False)
    assert out.startswith("Victorian study, gaslight")


def test_translation_is_case_insensitive(translator):
    lower = translator.translate("she stares", preserve_identity=False)
    upper = translator.translate("she STARES", preserve_identity=False)
    assert "intensity building" in lower
    assert "intensity building" in upper


def test_identity_anchor_is_optional(translator):
    with_anchor = translator.translate("she stares")
    without = translator.translate("she stares", preserve_identity=False)
    assert with_anchor.endswith("natural features preserved")
    assert not without.endswith("natural features preserved")


@pytest.mark.parametrize("function", list(CognitiveFunction))
def test_cognitive_flavor_is_appended(translator, function):
    out = translator.translate("she stares", cognitive_function=function,
                               preserve_identity=False)
    assert out.rstrip(".").split(", ")[-1]


# ---------------------------------------------------------------------------
# Grammar rules
# ---------------------------------------------------------------------------

def test_created_arc_has_no_doubled_preposition(translator):
    arc = translator.create_emotional_arc(
        subject="The young woman",
        emotion_start="quiet contemplation",
        emotion_end="inspired confidence",
    )
    assert "from from" not in arc
    assert "shifting from quiet contemplation to inspired confidence" in arc


@pytest.mark.parametrize("transition", NeuromorphicVocabulary
                         .EMOTIONAL_VOCABULARY["transitions"])
def test_vocabulary_transitions_never_double_a_preposition(transition):
    """Every listed transition must read correctly in the single-subject template."""
    arc = EmotionalArc(
        subject="she",
        initial_state="doubt",
        transition=transition,
        final_state="resolve",
        physical_manifestation="jaw setting",
    )
    line = GrammarRules.single_subject_arc("she", arc)
    for preposition in GrammarRules.TRAILING_PREPOSITIONS:
        assert f"{preposition} {preposition} " not in line + " "
    assert line.endswith("from doubt to resolve")


def test_inferred_physical_cue_follows_the_final_state(translator):
    arc = translator.create_emotional_arc("She", "doubt", "quiet realization")
    assert "eyes brightening" in arc


def test_two_subject_arc_keeps_both_subjects():
    a = EmotionalArc("she", "doubt", "giving way to", "resolve", "jaw setting")
    b = EmotionalArc("he", "pride", "softening toward", "respect", "gaze softening")
    line = GrammarRules.two_subject_sequential("she", a, "he", b)
    assert line.count("'s") == 2
    assert "resolve" in line and "respect" in line


def test_translate_for_video_reports_recommended_parameters(translator):
    result = translator.translate_for_video("woman smiles and nods",
                                            duration_seconds=4.0,
                                            num_subjects=1)
    assert result["recommended_steps"] == 24
    assert result["cognitive_function"] == CognitiveFunction.LANGUAGE.value
    assert "unfolding naturally over time" in result["prompt"]
    assert translator.translate_for_video("woman smiles", num_subjects=2)[
        "recommended_steps"] == 28


# ---------------------------------------------------------------------------
# Salience scoring
# ---------------------------------------------------------------------------

def test_emotional_words_are_not_counted_as_literal():
    """'motion' is a substring of 'emotion' -- it must not score as literal."""
    for word in ("emotion", "emotional", "emotionally"):
        scores = SalienceAnalyzer.estimate_salience(word)
        assert scores["literal_score"] == 0.0, word
        assert scores["emotional_score"] > 0.0, word


def test_literal_words_still_score_literal():
    for word in ("movement", "movements", "motionless", "position", "gesture"):
        scores = SalienceAnalyzer.estimate_salience(word)
        assert scores["literal_score"] > 0.0, word


def test_unrelated_word_is_not_a_literal_hit():
    """Prefix matching, not substring: 'reaction' is not the literal 'action'."""
    assert SalienceAnalyzer.estimate_salience("reaction")["literal_score"] == 0.0


def test_published_translation_clears_the_abstain_threshold():
    """The paper abstains below s_emotional = 0.3; the module's own
    translation of 'hand movement' must not fall under its own threshold."""
    emotional = NeuromorphicVocabulary.MOTION_TO_EMOTION["hand movement"]
    scores = SalienceAnalyzer.estimate_salience(emotional)
    assert scores["emotional_score"] >= 0.3


def test_salience_rises_after_translation(translator):
    literal = "Victorian woman portrait, subtle head movement, slight smile, blinking eyes"
    emotional = translator.translate(literal)
    before = SalienceAnalyzer.estimate_salience(literal)
    after = SalienceAnalyzer.estimate_salience(emotional)
    assert after["emotional_score"] > before["emotional_score"]
    assert after["recommended_steps"] < before["recommended_steps"]


def test_step_reduction_is_capped_at_25_percent():
    scores = SalienceAnalyzer.estimate_salience(
        "passion emotion warmth conviction determination hope fear intensity"
    )
    assert scores["recommended_steps"] >= 22
    assert float(scores["estimated_step_reduction"].rstrip("%")) <= 25.0
