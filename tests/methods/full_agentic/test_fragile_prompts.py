from hitl_pmp.core.practice_costs import ChargeFunction, PracticeCosts
from hitl_pmp.full_agentic.runner import FullAgenticRunner


def test_both_variants_receive_same_damage_mechanics_and_objective():
    costs = PracticeCosts(damage_contact=ChargeFunction(value=10))
    base = FullAgenticRunner.task_prompt(costs=costs)
    prompts = [
        FullAgenticRunner.variant_prompt(
            prompt=base, variant=v, fragile=True, mat_size=4, damage_cost=10
        )
        for v in ("original", "new-wording")
    ]
    for p in prompts:
        assert "C = C_R + w_H C_H + C_D" in p
        assert "cube is fragile" in p
        assert "4 m" in p and "outside the mat costs 10" in p
        assert "with the wall add no damage charge" in p
        assert "human placement are exempt" in p
        assert "hint: in order to minimize human cost" in p
    continuation = "When choosing a practice action, compare its immediate cost"
    assert continuation not in prompts[0]
    assert continuation in prompts[1]
