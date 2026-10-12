# Klein BF16 parameter-update characterization

This note records one bounded observation from historical FLUX.2 Klein
full-parameter experiments. It is not a repository-wide precision rule.

BF16 has seven explicit mantissa bits. Writing an update directly into a BF16
parameter can round a small step to zero when the step is below the local unit in
the last place. For a rough upper-bound calculation, an element with magnitude
`|w|` needs an update on the order of `|w| * 2^-7` to cross that spacing.

The historical Klein base scan found that 43.4% of elements were below the rough
`lr=1e-4` threshold and 5.0% below the `lr=1e-5` threshold. These figures are static
weight-distribution upper bounds, not measured update rates. The old claim that
roughly 4% or 0% of elements actually changed had no retained per-element probe and
must not be treated as verified evidence.

The durable conclusion is narrower:

- direct BF16 parameter storage can suppress small updates;
- an FP32 master parameter or FP32 trainable adapter avoids that storage floor;
- optimizer, precision, and topology must be qualified per model and recipe;
- a loss curve alone does not establish that the intended parameter set changed.

Any future precision claim should retain the script, environment, tensor inventory,
before/after displacement, and content-addressed result as a characterization receipt.
