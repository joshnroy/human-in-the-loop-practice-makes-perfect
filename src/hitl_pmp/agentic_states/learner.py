"""Joint code and natural-language state revisions from practice evidence."""

import shutil

from hitl_pmp.hybrid_skills.learner import SkillLearner
from hitl_pmp.step_protocol import StepFiles

from .budget import SharedBudget
from .graph import LanguageManifest

STATE_CONTRACT = """
SKILLS AND STATES CONTRACT (supersedes fixed structured initiation/effects):
You also own the abstract-state representation in submission/states.json. Supply
natural-language descriptions of observable, mutually distinguishable scene regions,
not Python predicates. A separate model classifies the same object-state observation
into one region, or unknown. Its inputs are your descriptions and the observation;
it has no privileged simulator labels or access to held-out outcomes.
The classical planner uses your graph to determine applicable skills and conditional
success/failure destinations. There is no hand-written PickPlannable gate or fallback.
The host retains classical Bayesian competence/learning-rate inference and measured
step/damage-cost forecasts. The task utility is the existing pick/toss/open deployment
success model. At deployment, a classical shortest-path search follows successful
robot edges to a goal region and replans after each classified observation. No human
skills are available at deployment. The simulator independently scores the real goal.

states.json must follow the JSON schema in states.schema.json. Choose your own cluster
IDs and descriptions, including physical goal regions. Each edge has source, action,
success and failure. Each destination may be a single cluster ID or a mapping of IDs
to positive conditional probabilities summing to 1. Success/failure destinations for
an edge must be disjoint so the observed endpoint determines the outcome. Success
must preserve the stated skill objective, not merely termination or code execution.
Edge sources determine applicability. Include both reset_cube_far and
reset_cube_and_bin_near, with their documented physical effects. These names bind to
host human tools; they cannot execute generated reset code or select arbitrary poses.
Robot action names remain PickCube, MoveToTossLocationAndToss and OpenGripper.
Do not write a classifier or call a model inside get_action. Write the complete skill
code AND state manifest before returning. Subsequent revisions may change either.
The one experiment budget covers coding plus runtime classification in practice and
evaluation; it does not replenish between calls. Classifier explanations from practice
are provided in evidence/; evaluation observations and explanations are withheld.
"""


class SkillsStatesLearner(SkillLearner):
    def __init__(self, *, on_accept, **kwargs):
        super().__init__(**kwargs)
        self.on_accept = on_accept
        self.revision_instruction = (
            " Read STATE_CONTRACT.md and revise submission/states.json together "
            "with the code when necessary. Resolve any unknown or ambiguous "
            "practice states using the recorded observations."
        )
        self.budget = SharedBudget(
            path=self.output / "model_budget.json", limit=kwargs.get("budget", 20), session="coding"
        )
        self.prompt = (
            self.prompt.replace(
                "The structured host retains its existing skill initiation/effect definitions.", ""
            )
            + STATE_CONTRACT
        )
        (self.output / "task_prompt.md").write_text(self.prompt)
        StepFiles.json(
            path=self.workspace / "states.schema.json", value=LanguageManifest.model_json_schema()
        )

    def prepare_evidence(self):
        super().prepare_evidence()
        target = self.workspace / "evidence" / "classifications"
        target.mkdir(exist_ok=True)
        for source in (self.output / "classifications" / "practice").glob(
            "*/sandbox/judgment.json"
        ):
            shutil.copyfile(source, target / (source.parent.parent.name + ".json"))
        (self.workspace / "STATE_CONTRACT.md").write_text(STATE_CONTRACT)

    def validate_revision(self):
        LanguageManifest.model_validate_json((self.submission / "states.json").read_text())

    def revise(self):
        # The process lock also excludes simultaneous evaluation classification,
        # so two Claude calls cannot each claim the remaining dollar budget.
        previous = self.env.bundle
        with self.budget.lock():
            try:
                super().revise()
                manifest = LanguageManifest.model_validate_json(
                    (self.submission / "states.json").read_text()
                )
                self.on_accept(manifest=manifest)
                StepFiles.json(
                    path=self.output / "revisions" / f"{self.calls:04d}" / "states.json",
                    value=manifest.model_dump(mode="json"),
                )
            except BaseException:
                self.env.bundle = previous
                raise
