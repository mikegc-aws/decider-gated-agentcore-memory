# Decider-gated AgentCore Memory session manager

> ### ⚠️ Experimental — proof of concept only
>
> This is a research spike, not a library. It is **not production code**, is not
> supported, has no stability guarantees, and is not affiliated with or endorsed by
> AWS or the Strands Agents project. Expect it to break against future versions of
> the upstream SDK, which it subclasses and depends on internals of.
>
> It also **depends on a decision model that has no public deployment** (see
> [Requirements](#7-requirements-and-running-it)). Without an endpoint of your own
> the gating path cannot run — though it fails open, so the agent still works.
>
> The quantitative claims come from small, self-labelled sets (20 utterances, 4
> queries) written by the same person who wrote the prompts being tested. The
> "100%" figures mean *no errors on this small set*, not *solved*. Don't build on
> the numbers without re-measuring.

A proof of concept: put a small decision model (`strands-decider-2b`, or Jev) in
front of Amazon Bedrock AgentCore Memory's long-term retrieval, inside the Strands
session manager, and answer two questions the stock integration never asks.

1. **Is a memory lookup worth making for this message at all?**
   "Hello" does not need a semantic search of the user's life story.
2. **Is each memory that came back actually relevant to the request?**
   `topK` returns *k* records whether or not *k* of them are any use.

Everything below was measured against a live AgentCore Memory store and the live
`strands-decider-g6` endpoint, not estimated.

---

## 1. What the stock integration does

`AgentCoreMemorySessionManager` (in `bedrock_agentcore.memory.integrations.strands`)
registers a `MessageAddedEvent` hook, `retrieve_customer_context`. On every user
text message it:

- fans out across the configured namespaces,
- runs `retrieve_memory_records` with a `topK` (default 10) per namespace,
- drops anything below a fixed `relevance_score` (default 0.2),
- splices the survivors into the user's message as a `<user_context>` block.

Confirmed still true in the current release — the newest SDK has been heavily
refactored (batching, `PersistenceMode`, metadata filters, async mode, bidi), but
the retrieval decision itself is unchanged: `session_manager.py:848-925`.

So the premise holds. The retrieval is blunt in both of the ways expected:

**It always fires.** There is no condition on the *content* of the message beyond
"is it user text". A greeting costs a full round trip to the memory service on the
critical path of the turn.

**A static score floor cannot fix the top-k problem.** This is the part worth
seeing concretely. Querying this PoC's own memory store for `"drink preference"`
(`scripts/check_memory.py`) returns, in one top-10:

```
0.403  The user's go-to coffee order is a decaf oat flat white.
0.384  The user dislikes sparkling water and prefers still water only.
0.380  The user strongly prefers four spaces (no tabs) for code indentation.
0.375  The user's favourite cuisine is Vietnamese.
0.370  The user is trying to cut down on caffeine.
...
0.357  The user enjoys pho and could eat it every week.
0.356  The user lives in Vancouver.
```

Look at rank 3. For a query about **drink preference**, *"prefers four spaces, no
tabs, for code indentation"* (0.380) outranks *"is trying to cut down on
caffeine"* (0.370). The embedding has no idea which of those bears on a drink.

All 19 records sit in a range of 0.356–0.496 — a spread of 0.140. No single
`relevance_score` keeps the top record and drops the bottom one, and the default of
0.2 keeps **everything**. Across an 8-turn conversation the stock manager injected
147 memory records totalling ~2,500 tokens, including on the turns "Hello!", "What
is 17 times 23?" and "ok cool".

## 2. What this fork changes

Exactly one method, `retrieve_customer_context`. Subclassing rather than copying
1,300 lines is deliberate: the write path (events, session restore, batching,
metadata) is not what is in question, and the gate is a policy change to a single
decision that should read as one.

```
          ┌─ gate: 1 noul question ─────────────────────────────┐
user msg ─┤  "does answering this need a stored personal fact?" │
          └─────────────┬───────────────────────────────────────┘
                 no ────┴──── yes
                  │            │
             skip entirely   retrieve top-k per namespace
             (no API call)     │
                               ├─ render records to prose
                               │
                          ┌────┴─ validate: N noul questions, ONE call ─┐
                          │  "is this memory useful for this request?"  │
                          └────┬────────────────────────────────────────┘
                               │
                      inject only what survives
```

The validation step batches on purpose. Extra questions about the same state are
nearly free (~47 ms for one, ~67 ms for seven), so checking ten memories is one
round trip, not ten.

## 3. The finding that actually mattered: wording, not model

The single biggest result here is not the architecture, it is that **the question
wording carries almost all of the quality**, and that it is cheap to measure.

`scripts/tune_gate.py` runs 5 candidate wordings against 20 labelled utterances,
batching all 5 into one call per utterance. On `strands-decider-2b`:

| gate wording | best accuracy | margin |
| --- | --- | --- |
| "answering this depends on a stored fact about the user" | **65%** | −0.279 |
| ...the same question with explicit `true`/`false` criteria | **100%** | +0.202 |
| "would a profile lookup change the answer" | 90% | −0.025 |
| inverted ("answerable without knowing the user") | 95% | −0.118 |
| 4-level `score` version | 95% | −0.049 |

65% is useless — barely better than always retrieving. The *same question* with the
boundary spelled out separates the two classes cleanly with a usable margin. The
`criteria` field is doing the work.

The same thing happened to the validator, and it bit before it was measured. The
first demo run used the obvious wording with an absolute 0.5 cut, and threw away
*"the user's go-to drink is a decaf oat flat white"* (p=0.484) in reply to *"I'd
like a drink, what should I get?"*. `scripts/tune_validator.py` shows why:

| validator wording | pooled margin | best acc | mean AUC | mean P@k |
| --- | --- | --- | --- | --- |
| plain | −0.022 | 80% | 0.835 | 69% |
| **criteria** | **+0.041** | **100%** | **1.000** | **100%** |
| topic-overlap | −0.401 | 77% | 1.000 | 100% |
| counterfactual ("would omitting it hurt") | −0.137 | 74% | 0.490 | 29% |
| graded `score` | −0.105 | 91% | 0.935 | 85% |

Two things to take from that table:

- **Separation and ranking fail independently.** The topic-overlap wording ranks
  perfectly (AUC 1.000) but is hopeless at any absolute threshold (margin −0.400).
  If you only measure one of these you will pick the wrong wording.
- The counterfactual framing *inverts* (AUC 0.496, P@k 29%) — it is actively worse
  than random. Plausible-sounding prompts can be anti-correlated, which is a good
  argument for measuring rather than reasoning about wording.

## 4. A second real bug: JSON records skew the scores

AgentCore's strategies do not return the same shape. `SEMANTIC` gives prose
("The user is vegetarian."); `USER_PREFERENCE` gives a JSON blob:

```json
{"context":"Ordering a drink or beverage","preference":"Dislikes sparkling water","categories":["beverages"]}
```

Upstream injects that verbatim. It survives contact with a large model, but it
measurably skews the validator: the raw-JSON records scored *systematically
higher* regardless of relevance, so a threshold tuned on prose let JSON false
positives through — "always books an aisle seat" was kept for a question about what
to drink. `records.py` flattens both strategies to one prose style first
("When ordering a drink or beverage: Dislikes sparkling water"). That alone took
the drink query from 10 memories kept to 7, dropping exactly the wrong ones
(aisle seat, four-spaces, YVR, Vietnamese).

This rendering is applied in **both** arms of the comparison, so the measured
difference stays attributable to gating alone.

## 5. Results

8-turn mixed conversation, same memory store, Claude Sonnet 4.5 as the agent.
`decider=None` in arm A reproduces upstream behaviour exactly.

| | A: upstream | B: gated (2b) | B: gated (Jev) |
| --- | --- | --- | --- |
| memory retrievals | 8 | **3** | **3** |
| records injected | 147 | **25** | **16** |
| context injected | ~2,471 tok | **~369 tok** | ~253 tok |
| memory service time | 4,230 ms | **1,900 ms** | 1,745 ms |
| decider calls | 0 | 14 (4,867 ms) | 14 (7,210 ms) |

The gate closed on exactly the right 5 turns — "Hello!", "What is 17 times 23?",
"Thanks!", "Can you explain what a bloom filter is?", "ok cool" — and opened on the
3 that needed memory. **Answer quality is unchanged**: both arms recommend the decaf
oat flat white, both remember the aisle seat and YVR, both suggest vegetarian pho
with the peanut warning. The gated arm gets there on ~15% of the context.

### The honest caveat on latency

Wall-clock is a wash here (29.9 s gated vs 29.3 s baseline), and that is not a win
worth claiming. The decider is being called **from a laptop outside `us-west-2`**,
so each call costs ~350 ms of round trip against ~50–70 ms of actual server-side
work. 14 calls of avoidable network is roughly the 2.3 s saved on the memory
service. Co-located in-region the arithmetic turns positive, but that is an
inference from the endpoint's documented latency, not something measured here.

So the defensible wins from this PoC are **the 85% cut in injected context** and
**5 of 8 memory API calls avoided**. Treat the latency case as unproven until it is
run in-region.

### Fail-open, and why it is reported

Both the gate and the validator fail open: if the decider errors, the manager
retrieves and keeps everything, degrading to upstream behaviour rather than
blinding the agent. That is the right default, but it hides failure — during
development a misconfigured endpoint name made *every* decision error, and the run
still produced correct answers, so the stats just looked oddly ineffective. Hence
`GateStats.decider_errors`, which is printed loudly when non-zero:

```
  !! decider errors          16  (failed open -- results below reflect UNGATED behaviour)
```

If you see gate decisions counted but neither opened nor closed, that is what
happened.

## 6. Layout

```
src/decider_memory/
  session_manager.py   the fork: overrides retrieve_customer_context, nothing else
  deciders.py          one interface, two backends (SageMaker/Triton, Jev/OpenRouter)
  config.py            tuned questions, per-backend thresholds, stats
  records.py           flatten AgentCore record shapes to prose
scripts/
  setup_memory.py      create + seed the AgentCore Memory resource
  check_memory.py      show stored records and how flat the scores are
  tune_gate.py         measure gate wordings against labelled utterances
  tune_validator.py    measure validator wordings (separation AND ranking)
demo.py                side-by-side upstream vs gated
tests/test_gate.py     16 tests, stubbed decider, no network
```

## 7. Requirements, and running it

**You need a decision model.** Two backends are implemented:

- `strands-decider-2b` on a SageMaker real-time endpoint (Triton). **There is no
  public deployment of this model.** If you have your own endpoint, point the code
  at it with `DECIDER_ENDPOINT` / `DECIDER_REGION`.
- **Jev**, via OpenRouter's `/decisions` endpoint — this one *is* publicly
  reachable with an `OPENROUTER_API_KEY`, so `--decider jev` is the path most
  people can actually run.

You also need AWS credentials with Bedrock AgentCore Memory and `bedrock-runtime`
access in `us-west-2`, and the agent itself calls Claude Sonnet 4.5 on Bedrock.

Only the tests run with no credentials at all.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q                 # no credentials needed

cp .env.example .env                          # add OPENROUTER_API_KEY for Jev

.venv/bin/python scripts/setup_memory.py      # creates + seeds; writes .memory_id
                                              # wait a few minutes for extraction
.venv/bin/python scripts/check_memory.py "drink preference"

.venv/bin/python demo.py --decider jev        # the runnable-by-default path
.venv/bin/python demo.py                      # strands-decider-2b (needs an endpoint)
.venv/bin/python demo.py --quiet              # hide the per-turn trace

.venv/bin/python scripts/tune_gate.py jev
.venv/bin/python scripts/tune_validator.py jev
```

`setup_memory.py` creates an AgentCore Memory resource with two strategies
(`SEMANTIC` and `USER_PREFERENCE`) and seeds it with five short conversations, then
AgentCore's extraction pipeline takes a few minutes to produce the ~19 long-term
records the demo queries. **This creates billable AWS resources** and the script
does not delete them — remove the memory when you are done.

The seeded persona (`demo_user`) is **entirely fictional**. The preferences, the
dietary details and the dog are invented demo data, chosen to span several
unrelated areas of a life so that semantic top-k has a realistic chance of
returning the wrong one.

## 8. Licence

MIT — see [LICENSE](LICENSE). Not affiliated with AWS or the Strands Agents project.

## 9. Where this would go next

- **Run it in-region** and settle the latency question properly.
- **Route, don't just gate.** The gate is one `noul`; a `choice` question in the
  same call could pick *which* namespace is worth searching, so a drink question
  never searches travel memories. That is free — it is the same round trip.
- **Thresholds are per-backend and need owning.** Across repeated tuning runs,
  rankings were stable for both models (per-query AUC 1.000 every time) but Jev's
  *absolute* values drifted enough to move its best pooled threshold, while
  `strands-decider-2b`'s held. 2b is the tuned path here; the Jev numbers in
  `BACKEND_DEFAULTS` are a reasonable start, not a measured optimum. A relative
  rule (keep what is close to the best for *this* query) would likely be more
  robust than an absolute cut — it was tried against the tuning data and lost to
  the absolute threshold on 2b, so it is not in the code, but with a drifting
  backend it is the obvious next thing.
- **The labelled sets are tiny** — 20 utterances and 4 queries, written by the same
  person who wrote the prompts. The 100% figures mean "no errors on this small
  set", not "solved". Anything load-bearing needs a bigger, independently labelled
  set and a held-out split.
