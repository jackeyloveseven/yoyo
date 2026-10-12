# Architecture decision records

Create an ADR before a core architecture change adds more than 500 lines. Use a
short numbered filename such as `0001-objective-plugin-lifecycle.md` and include:

1. context and the concrete decision;
2. alternatives and consequences;
3. affected stable contracts;
4. old files or interfaces removed in the same migration;
5. verification and rollback boundaries.

An ADR is not a status report or implementation diary. Update the stable contract
after the decision lands and keep mutable rollout evidence in receipts.
