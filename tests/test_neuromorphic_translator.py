import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(MODULE_DIR))

from neuromorphic_prompt_translator import (  # noqa: E402
    CognitiveFunction,
    EmotionalArc,
    GrammarRules,
    NeuromorphicTranslator,
    SalienceAnalyzer,
)


def test_translate_replaces_longer_motion_phrases_before_generic_moves():
    translator = NeuromorphicTranslator()

    result = translator.translate("woman has head movement, slight smile, and blinks")

    assert "subtle shift in attention" in result
    assert "knowing smile forming" in result
    assert "eyes softening with thought" in result
    assert "with expressive presence" in result
    assert "natural features preserved" in result
    assert result[0].isupper()


def test_translate_supports_cognitive_flavors_and_identity_toggle():
    translator = NeuromorphicTranslator()

    result = translator.translate(
        "person moves through the room",
        cognitive_function=CognitiveFunction.SPATIAL,
        preserve_identity=False,
    )

    assert "shifting with purposeful energy" in result
    assert "moving through the space with purpose" in result
    assert "natural features preserved" not in result


def test_create_emotional_arc_infers_physical_cues():
    translator = NeuromorphicTranslator()

    arc = translator.create_emotional_arc(
        subject="The inventor",
        emotion_start="uncertainty",
        emotion_end="quiet determination",
    )

    assert arc.startswith("The inventor's jaw setting with quiet resolve")
    assert "uncertainty" in arc
    assert arc.endswith("quiet determination")


def test_grammar_rules_format_single_and_anchored_arcs():
    arc = EmotionalArc(
        subject="she",
        initial_state="hesitation",
        transition="softening",
        final_state="trust",
        physical_manifestation="gaze opening",
    )

    assert GrammarRules.single_subject_arc("Sophia", arc) == (
        "Sophia's gaze opening, softening from hesitation to trust"
    )
    assert GrammarRules.anchored_emotional("Sophia", ["period dress", "brass lab"], arc) == (
        "Sophia with period dress, brass lab, gaze opening, softening trust"
    )


def test_translate_for_video_adjusts_long_duration_and_multi_subjects():
    translator = NeuromorphicTranslator()

    one_subject = translator.translate_for_video("woman smiles", duration_seconds=4.5)
    two_subjects = translator.translate_for_video(
        "woman smiles and man nods",
        duration_seconds=2.0,
        num_subjects=2,
    )

    assert "gradual emotional progression" in one_subject["prompt"]
    assert one_subject["recommended_steps"] == 24
    assert "each figure with their own emotional presence" in two_subjects["prompt"]
    assert two_subjects["recommended_steps"] == 28
    assert "warped face" in two_subjects["negative"]


def test_salience_analyzer_scores_emotional_language_above_literal_motion():
    emotional = SalienceAnalyzer.estimate_salience(
        "quiet determination and fierce conviction with warmth spreading"
    )
    literal = SalienceAnalyzer.estimate_salience(
        "physical movement position angle direction speed"
    )

    assert emotional["emotional_score"] > literal["emotional_score"]
    assert emotional["recommended_steps"] < literal["recommended_steps"]
    assert literal["literal_score"] > emotional["literal_score"]
