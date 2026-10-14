"""The Full-Agentic Tossing3D domain observes and specifies exactly as the direct bridge.

It builds a Tossing3D simulator, so it lives here: CI renders Tossing3D only in its
own job (OSMesa), and a sim test collected by the unit job has no GL context.
"""

import importlib.util
import json

import pytest

from hitl_pmp.full_agentic.domains import Tossing3DFullAgentic

HAS_KINDER = importlib.util.find_spec("kinder") is not None


@pytest.mark.skipif(not HAS_KINDER, reason="needs KINDER")
@pytest.mark.xdist_group("full_agentic_tossing3d_sim")
class TestTossing3DBridgeUnchanged:
    def test_domain_bridge_observes_and_specifies_exactly_as_the_direct_bridge(
        self,
    ) -> None:
        from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge

        env = Tossing3DFullAgentic.make_environment(options={}, human_seed=0)
        try:
            env.hard_reset()
            direct = Tossing3DAgenticBridge(
                env=env, observation_mode="object_state", step_limit=2**63 - 1
            )
            via_domain = Tossing3DFullAgentic.make_bridge(env=env, practice=True)
            assert via_domain.step_limit == direct.step_limit
            assert json.dumps(via_domain.observe()) == json.dumps(direct.observe())
            assert json.dumps(via_domain.robot_spec()) == json.dumps(direct.robot_spec())
            evaluation = Tossing3DFullAgentic.make_bridge(env=env, practice=False)
            assert evaluation.step_limit == 1000  # the bridge's own default, as before
        finally:
            env.close()
