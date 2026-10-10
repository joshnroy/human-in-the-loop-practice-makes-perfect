"""Immutable skill/state snapshots and language-based held-out deployment."""

import pickle
from pathlib import Path

import numpy as np

from hitl_pmp.agentic_runtime.sandbox import SandboxSettings
from hitl_pmp.core.method.types import LabeledAction
from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge
from hitl_pmp.hybrid_skills.artifacts import SkillBundle
from hitl_pmp.hybrid_skills.execution import prepare_problem
from hitl_pmp.step_protocol import StepEvaluation

from .classifier import LanguageClassifier
from .graph import LanguageManifest, StateGraph


class StatesDeployment:
    def __init__(self, *, env, manifest, classifier):
        self.env, self.graph, self.classifier = env, StateGraph(manifest=manifest), classifier

    def get_task_policy(self, *, task):
        def policy(state):  # noqa: PLR0917 -- Policy protocol is positional
            observation = Tossing3DAgenticBridge(
                env=self.env, observation_mode="object_state"
            ).observe()
            cluster = self.classifier.classify(
                manifest=self.graph.manifest, observation=observation
            ).cluster_id
            name = self.graph.deployment_action(cluster=cluster)
            if name is None:
                return LabeledAction(
                    action=self.env.noop_action(), label="No language-graph deployment path"
                )
            identifier = {
                "PickCube": self.env.pick_cube_id,
                "MoveToTossLocationAndToss": self.env.move_to_toss_location_and_toss_id,
                "OpenGripper": self.env.open_gripper_id,
            }[name]
            return LabeledAction(action=np.array([identifier, 0, 0, 0, 0], dtype=float), label=name)

        return policy


class StatesSnapshot:
    @staticmethod
    def encode(*, method):
        if method.manifest is None or method.env.bundle is None:
            raise RuntimeError("Both accepted states and code are required before evaluation")
        return pickle.dumps(
            dict(
                manifest=method.manifest.model_dump(),
                bundle=method.env.bundle.model_dump(),
                classifier=method.classifier_configuration,
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    @staticmethod
    def restore(*, raw, env, provider):
        payload = pickle.loads(raw)
        env.bundle = SkillBundle.model_validate(payload["bundle"])
        configuration = payload["classifier"]
        classifier = None
        if configuration:
            classifier = LanguageClassifier(
                settings=SandboxSettings.model_validate(configuration["settings"]),
                output=Path(configuration["output"]),
                budget_path=Path(configuration["budget_path"]),
                budget_limit=configuration["budget_limit"],
                phase="evaluation",
            )
        return StatesDeployment(
            env=env,
            manifest=LanguageManifest.model_validate(payload["manifest"]),
            classifier=classifier,
        )


class StatesEvaluation:
    @staticmethod
    def run(**kwargs):
        return StepEvaluation.run(
            **kwargs, prepare_problem=prepare_problem, restore_method=StatesSnapshot.restore
        )
