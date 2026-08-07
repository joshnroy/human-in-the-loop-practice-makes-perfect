# Tossing Room, in words

This file is the payload of `--method pure-agent-author --pure-agent-prompt-arm
described`. It is a natural-language account of the world, appended to the symbolic
layer the minimal arm already gets.

**What it deliberately does and does not say.** It names the *mechanism* — that a throw
needs a particular force, and that the force depends on the item's weight — because that
is the analogue of the reference notebook naming `Pendulum-v1`, which lets an agent
recall the pendulum's dynamics. It does **not** give the coefficients, the intercept or
the tolerance; recovering those is the problem, and handing them over would make this an
oracle rather than a hint. Nothing below is a number the environment uses.

---

A robot works in a row of seven rooms, numbered 0 to 6, laid out left to right. It can
step to an adjacent room, one room at a time.

The robot starts in room 3. Room 3 also holds a limitless pile it can take items from.
The pile issues two kinds of item: trash and recycling. The robot has one hand, so it can
carry one item at a time and must pick up before it can throw.

There is a trash bin in room 6 and a recycling bin in room 1. Each bin takes only its own
kind of item. Each bin holds at most one item at a time, and each has a button beside it,
in the same room, that empties it. A throw at a bin that is already full is refused
outright: nothing happens and the item stays in the robot's hand.

**The corridor is one-way at one point.** There is a ledge between room 2 and room 3. The
robot can step left across it, from room 3 into room 2, but never right, from room 2 back
into room 3. Everything else in the corridor is two-way. Since the pile is in room 3, and
room 3 is unreachable once the robot has gone left past it, the robot gets exactly one
recycling item per episode: it picks up, walks left across the ledge to room 1, and
whatever happens to that throw is final. Trash is a round trip and can be retried.

To throw, the robot must be standing in the bin's own room, holding an item of that bin's
kind, and it chooses a **force** — a single continuous number.

**A throw always releases the item, whether or not it lands.** There is no way to pick a
thrown item back up; an item that missed is simply gone, and getting another one means
returning to the pile. This is what makes a miss cost something.

A throw lands in the bin only if the chosen force is close enough to the force that
particular throw needed. "Close enough" means within a fixed tolerance, the same for
every throw.

**The force a throw needs is not written anywhere in the state.** What is in the state is
its cause: the weight of the item being thrown. Heavier items need more force, and the
relationship between weight and required force is fixed — the same relationship for
trash and for recycling, so anything learned about one transfers to the other.

**An item's weight is drawn afresh every time the robot picks one up**, from a fixed
range, so consecutive attempts are at different weights. An item's weight before it has
been picked up is a placeholder and means nothing.

Goals come in three families: get a trash item into the trash bin, get a recycling item
into the recycling bin, or empty both bins. Emptying both bins is an ordering problem: the
recycling bin's button is behind the one-way ledge, so the trash button must be pressed
first.
