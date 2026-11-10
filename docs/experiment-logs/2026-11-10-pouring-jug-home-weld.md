# Anchor the parked kitchen jug

The all-fixes Della run, EES p=0 seed 0 on `d9464fd84`, ended at 63,223 practice
steps after a successful dumpling drop rotated the jug about 17 degrees. The next
pick remained applicable, but its drop required the now-false `PitcherHome` atom.
At p=0 the subsequent 204 human attempts could not repair the scene.

The kitchen now anchors the parked jug with a switchable world weld, like the
skillet's support weld. Its free joint and qpos layout remain intact. The weld
captures the measured pose without assigning positions or velocities. A verified
two-finger handle grasp releases the support before lifting. Placement re-engages
it after the jug lands at home, before the empty gripper withdraws. A successful
human hand-off also re-engages it. Grasp, carry and pouring retain their existing
physical handle contact; a gripper weld is a separate follow-up.

Anchoring requires home alignment and a height within 3 mm of support. A hovering
or misaligned placement is rejected, rather than converted to success. The new
constraint is disabled during pickup, carry and pouring. No planner, prior,
cost, pour parameter or human-success probability changes accompany it.

## Verification

The saved pre-drop state from practice step 62,105 reproduces the failure on the
old controller: the drop succeeds but the jug is no longer home. The same test
passes with the support weld. The fixture retains its run identity and recording
SHA-256. Additional native tests cover a force applied to the parked jug,
pick-pour-place releasing and restoring the weld, and a successful human reset
from a held jug. A guard test rejects hovering and misaligned placement poses.

The draft is based on the exact all-fixes launch branch so the replacement sweep
retains the earlier recovery fixes. Its Della launch uses a fresh result directory
and the existing 30-row configuration; prior records remain separate.
