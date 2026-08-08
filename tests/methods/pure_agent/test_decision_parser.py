"""What the parser forgives and what it refuses.

The split is the whole design: packaging is forgiven because rejecting a fenced block
would measure the prompt's ability to suppress prose rather than the agent's ability to
choose an action; content is refused because repairing an illegal decision would invent a
choice the agent did not make.

Cases are looped rather than `parametrize`d because every parameter in this project is
keyword-only (ruff PLR0917) and `parametrize` injects positionally. The loop also puts the
offending case into the assertion message, which a parametrized id does less directly."""

import pytest

from hitl_pmp.methods.pure_agent.decision_parser import DecisionParseError, DecisionParser


def parse(*, text, num_skills=3, param_dims=(0, 1, 2)):
    """Three applicable skills taking 0, 1 and 2 parameters -- enough to exercise every
    width check without any of them coinciding."""
    return DecisionParser.parse(text=text, num_skills=num_skills, param_dims=list(param_dims))


def test_the_object_is_found_wherever_it_sits():
    """Every one of these is a shape a real reply takes. Rejecting any would measure the
    prompt's ability to suppress prose rather than the agent's ability to choose."""
    for text in [
        '{"skill_index": 1, "params": [3.5]}',
        '  {"skill_index": 1, "params": [3.5]}  \n',
        '```json\n{"skill_index": 1, "params": [3.5]}\n```',
        'Looking at the weight, I will throw.\n{"skill_index": 1, "params": [3.5]}',
        '{"skill_index": 1, "params": [3.5]}\nThat should land it in the bin.',
    ]:
        choice = parse(text=text)
        assert choice.skill_index == 1, text
        assert choice.params == (3.5,), text


def test_the_last_object_wins_because_the_agent_settles_after_it_reasons():
    """An earlier brace in a narrated reply is usually a hypothesis the agent went on to
    reject; the one it acts on comes last."""
    text = (
        'I considered {"skill_index": 0, "params": []} but the item is heavy, so: '
        '{"skill_index": 1, "params": [9.0]}'
    )
    assert parse(text=text).skill_index == 1


def test_a_nested_object_is_not_truncated():
    """Balanced-brace scanning rather than a regex: a non-greedy regex silently cuts a
    nested object short and then reports a JSON error that names the wrong problem."""
    text = '{"reasoning": {"note": "heavy"}, "skill_index": 1, "params": [4.0]}'
    assert parse(text=text).params == (4.0,)


def test_a_quoted_brace_does_not_desynchronise_the_scan():
    text = '{"note": "use { carefully", "skill_index": 1, "params": [4.0]}'
    assert parse(text=text).skill_index == 1


def test_a_reply_that_names_no_decision_is_refused():
    for text in [
        "",
        "   ",
        "I am not sure what to do here.",
        '{"params": [1.0]}',
        '{"skill_index": "one", "params": [1.0]}',
        '{"skill_index": true, "params": [1.0]}',
    ]:
        with pytest.raises(DecisionParseError):
            parse(text=text)


def test_an_index_outside_the_applicable_set_is_refused_rather_than_clamped():
    """The applicable set is what makes an illegal action unrepresentable. Clamping would
    hand that guarantee back."""
    for index in [-1, 3, 99]:
        with pytest.raises(DecisionParseError):
            parse(text=f'{{"skill_index": {index}, "params": []}}')


def test_a_parameter_vector_of_the_wrong_width_is_refused():
    with pytest.raises(DecisionParseError):
        parse(text='{"skill_index": 2, "params": [1.0]}')  # skill 2 takes 2
    with pytest.raises(DecisionParseError):
        parse(text='{"skill_index": 0, "params": [1.0]}')  # skill 0 takes none


def test_a_non_finite_or_non_numeric_parameter_is_refused():
    with pytest.raises(DecisionParseError):
        parse(text='{"skill_index": 1, "params": [1e999]}')
    with pytest.raises(DecisionParseError):
        parse(text='{"skill_index": 1, "params": ["3.0"]}')


def test_an_unparameterized_skill_accepts_an_omitted_or_null_params():
    """`[]`, `null` and an absent key are the same statement; rejecting one would be a
    parser preference rather than a real distinction."""
    assert parse(text='{"skill_index": 0}').params == ()
    assert parse(text='{"skill_index": 0, "params": null}').params == ()
    assert parse(text='{"skill_index": 0, "params": []}').params == ()


def test_an_integer_parameter_is_accepted_as_a_float():
    assert parse(text='{"skill_index": 1, "params": [4]}').params == (4.0,)


def test_the_error_message_names_the_problem():
    """These strings land in the ledger and are the whole diagnosis for a run whose
    decisions went wrong, so they are asserted rather than left to drift."""
    with pytest.raises(DecisionParseError, match="outside the applicable set"):
        parse(text='{"skill_index": 7, "params": []}')
    with pytest.raises(DecisionParseError, match="takes 2"):
        parse(text='{"skill_index": 2, "params": []}')
