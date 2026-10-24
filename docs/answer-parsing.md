# Answer extraction policy

`final-answer-v2` is the parser policy used for newly scored predictions. It requires
exactly one case-sensitive `FINAL_ANSWER:` marker and parses only the text after that
marker. The suffix must contain exactly one supported numeric value (US thousands
commas, optional sign, decimal, or exponent), with surrounding whitespace allowed.
One trailing sentence-ending period is accepted only when the value is an integer,
so `180.` means 180 while `3.5` remains a decimal and `3.5.` is rejected. Units,
lists, fractions, prose after the value, malformed grouping, missing markers, and
multiple markers are rejected. Numbers in the reasoning before the marker are never
used as a fallback. The same policy applies to every language and model condition.

The policy version is included in score records and metrics. Historical results are
not rewritten: re-score their preserved raw JSONL into a new output directory to apply
this policy. `final-answer-v1` required the marker on its own terminal line and did
not accept terminal integer punctuation.
