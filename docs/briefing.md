# What a Loop Sees: progress briefing

**For:** Prof. Garg and Dr. Saini
**Date:** 3 September 2026
**Status:** day 4 of a 16 week plan, stage 3 of 13

Short version: the foundations are built and tested, we have one genuine
finding about how these models learn, we caught three problems that would
have been expensive later, and we do not yet have results for any of the
project's four main questions. That is roughly where the plan expects us to
be in week one.

---

## 1. The question

A computer model that looks at a picture can be built to "think in loops":
run the same reasoning step over and over before answering. Nobody knows
what actually determines how many loops it needs.

There are two candidates. It might need more loops when the reasoning chain
is longer, a question with four linked steps rather than one. Or it might
need more loops when the picture is busier, twenty objects rather than five.
These are different things, and the answer matters, because every system of
this kind currently decides when to stop looping using a rule of thumb that
nobody has checked against ground truth.

The reason nobody has answered it is that on real photographs you cannot
know how much reasoning the correct answer required. We avoid that by
generating our pictures from recipes, so we know the answer and we know
exactly how many steps it takes to reach it.

---

## 2. What we found

Our main puzzle shows a shape and a row of instructions, rotate, flip,
recolour, and asks what the shape looks like at the end. Order matters:
rotating then flipping gives a different result from flipping then rotating.
There are 48 possible answers, so random guessing scores about 2 in 100.

**The model first learns the part of the puzzle where order does not matter,
learns it perfectly, and stays completely ignorant of the part where order
does matter.**

An analogy. It is like a student who has worked out that an even number of
flips always returns you to where you started, and can tell you that
reliably every time, while never learning the actual sequence of moves. They
know something true about the answer without being able to work the answer
out.

We measured this directly. Three facts about the answer can be worked out
without knowing the order of the instructions. The model gets two of them
exactly right every single time, and is at pure guesswork on everything
else.

| What we measured                | Model  | Guessing |
|---------------------------------|--------|----------|
| Order-free fact one             | 100%   | 50%      |
| Order-free fact two             | 100%   | 50%      |
| Order-free fact three           | 49.8%  | 50%      |
| The full answer                 | 16%    | 2%       |

This matters because the whole project is about step by step reasoning, and
the finding says the model reaches first for the shortcut that skips the
steps entirely. Two independent training runs now agree, so it is not a
fluke of one attempt.

One loose end, stated plainly: the model scores about twice as well as those
two facts can account for, so it has picked up something further that we
have not yet identified. We will not put this in the paper until we know
what it is.

---

## 3. Three problems we caught

Catching these in week one rather than in month three is the good outcome.

**The model was scoring above chance while doing nothing at all.** On the
harder versions of the puzzle, scores looked mildly encouraging, roughly
double random guessing. We then checked something more revealing than the
score, namely how many different answers the model ever gives, and found it
was producing the same one or two answers out of forty eight for every
picture in the dataset. Like a student who answers "C" to everything and
gets a few right. Had we trusted the score, our central measurement would
have been built on a model that was doing nothing, and the resulting graph
would have looked perfectly reasonable.

**One of the planned validity checks could never have passed.** The project
plan required a check to score no better than random guessing. That is
mathematically impossible for this kind of puzzle: a method that ignores
instruction order entirely can still score thirty one times better than
guessing, purely from the structure of the problem. We replaced it with a
check that is achievable and meaningful, and recorded the change and the
reason. Worth noting that the plan's own suggested test would have reported
"no shortcut here" while a large shortcut sat unmeasured.

**We nearly abandoned a task that was working.** We first trained for 20,000
rounds, saw no learning at all, and concluded the puzzle was broken. Running
it for 200,000 rounds took it from useless to perfect. The improvement was
not gradual: the model sat completely flat for 50,000 rounds and then jumped
from 16% to 88% in under 8,000. We had stopped roughly four times short of
the point where it starts working.

---

## 4. What we are waiting on

Two long training runs, about twenty hours from now. They settle one
question: is the two step version of the puzzle solvable given enough
training, or do we need to simplify the puzzle?

The current signal is not encouraging. The model has been stuck at the same
level for 270,000 rounds. When the one step version got stuck, it broke
through after 48,000. This one is already stuck five and a half times
longer.

If the runs finish stuck, we simplify the puzzle, and we will be doing that
on evidence rather than on a hunch. That is the reason for running them
rather than deciding now.

---

## 5. Practical notes

**The shared cluster is heavily contended.** At one point 94 of the 96
processor cores on the machine were taken while three of its eight graphics
cards sat completely idle. The cards were idle because nothing had
processors left to feed them. We have restructured how we submit work and
now get roughly twice as much running inside the same allocation.

**We have cut the model size comparison from four sizes to two.** This
fallback was written into the plan in advance for exactly this situation.
The paper will say so plainly rather than presenting two points as though
they were four.

**Everything is on the record.** Every decision, every failure and every
number is written down and traceable to the run that produced it. Where we
changed a plan, both the original and the reason are preserved, so nothing
has to be reconstructed from memory at writing time.

---

## 6. What we do not have

Being explicit, since the account above is mostly groundwork and problems
found.

- No results for any of the project's four main questions.
- No graph of the central measurement worth showing.
- The finding in section 2 has the unresolved loose end noted there.

The timetable still has results landing well before the submission deadline.
The failures so far are the cheap kind, found before anything was built on
top of them.

---

*Full technical record in `docs/findings.md`, `docs/decisions.md` and
`docs/progress.md`. Happy to walk through any of it.*
