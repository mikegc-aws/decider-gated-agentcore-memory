# Should the agent even look that up?

> All the code, both tuning harnesses and the script that builds and seeds the memory store
> are on GitHub: **[github.com/mikegc-aws/decider-gated-agentcore-memory](https://github.com/mikegc-aws/decider-gated-agentcore-memory)**.
> This post is the writeup. The repo lets you re-run any number in it.

*A caveat up front. What follows is one afternoon's worth of measurement on a synthetic
memory store, with labels I wrote myself. I have tried to be quantitative where I can, but
the sets are small enough that you should read direction and not magnitude. I should also
say plainly that I am not a neutral party here, so discount accordingly.*

I am one of the authors on the [strands-decider launch
post](https://strandsagents.com/blog/introducing-strands-decider/), along with Marc Brooker
and Fabio Nonato de Paula. In that post we list the places a decision model earns its keep,
and memory is one of the words in that list.

Lists are cheap. What a list does not tell you is whether the idea survives contact with a
real retrieval path, what the question has to look like before it works, or how much of the
win is still there once you measure it rather than assert it. Writing "memory" in a use case
list took me about four seconds. So I went and built the memory one properly, to find out
whether we had earned the word.

A quick description for anyone who has not met it. `strands-decider` is a 2B open source
decision model, and it does not generate text. You hand it some state and one or more typed questions, and it returns
probabilities with a calibrated confidence, and extra questions about the same state cost
almost nothing. Under the hood it is a Qwen3.5-2B torso with the language model head
replaced by a small pointer head and tuned with a LoRA, which Marc wrote up in detail
[here](https://brooker.co.za/blog/2026/09/28/engineering-system-one.html). It runs locally.
`pip install strands-decider`, median latency around 115ms on an RTX 3090 and about 153ms on
an M3 MacBook, weights and training data on
[HuggingFace](https://huggingface.co/StrandsAgents), code at
[strands-labs/strands-decider](https://github.com/strands-labs/strands-decider).

Here is the question I started with. When an agent has long term memory, something has to
decide what to recall on each turn. Does that decision deserve a model?

## What Strands does today

`AgentCoreMemorySessionManager`, in the `bedrock-agentcore` SDK, wires Strands up to Amazon
Bedrock AgentCore Memory. It writes each message to the memory service as an event, and it
registers a `MessageAddedEvent` hook, `retrieve_customer_context`, for the way back in.

That hook is all of the retrieval logic. On a user text message it fans out over the
configured namespaces, runs a semantic search with a `topK`, drops anything under a fixed
`relevance_score`, and splices what is left into the user's message inside a
`<user_context>` block.

It is about seventy lines, and to be fair to it, it is doing a sensible default thing. But it makes two decisions implicitly, and I think both are more interesting than
they first look.

*Decision one: should we retrieve at all?* The hook's only condition is "is this user
text". So a greeting runs a semantic search over the user's stored history before the agent
says hi.

*Decision two: which of the returned records belong in the prompt?* The answer today is "the
ones above a fixed score". Whether that works depends entirely on how the scores are
distributed, which is worth actually looking at.

## The score distribution does not cooperate

I seeded a store with five short conversations about an invented person, covering drinks,
food, travel, work and a dog. AgentCore's extraction produced 19 long term records. Here is
what a search for "drink preference" returns, best first:

```
0.403  The user's go-to coffee order is a decaf oat flat white.
0.384  The user dislikes sparkling water and prefers still water only.
0.380  The user strongly prefers four spaces (no tabs) for code indentation.
0.375  The user's favourite cuisine is Vietnamese.
0.370  The user is trying to cut down on caffeine.
...
0.356  The user lives in Vancouver.
```

Here is the whole result, coloured by whether the record is actually about what the user
drinks.

![All 19 long term memory records returned for the query "drink preference", as a horizontal bar chart sorted by semantic similarity score. Records about drinks and records about unrelated subjects are interleaved through a narrow band from 0.356 to 0.496, and the default relevance_score floor of 0.2 sits far to the left of every record.](https://blog.mikegchambers.com/assets/images/decider-gated-memory/score-spread.jpg)

The indentation preference scores above the note about cutting down on caffeine, for a
question about drinks. I do not think that is a defect in the embedding. It is being asked
"what here is about this subject", and it answers that; "which of these should the agent say
right now" is a different question that nobody asked.

The part that matters for the fixed threshold is the spread. All 19 records land between
0.356 and 0.496, and the relevant and irrelevant ones are interleaved through that band.
There is no value of `relevance_score` that keeps the top record and drops the bottom one.
The default is 0.2, so in practice everything gets through. Over an eight turn conversation
the stock manager put 147 records, roughly 2,500 tokens, into the prompt, including on the
turns "Hello!", "What is 17 times 23?" and "ok cool".

So, two candidate questions for a classifier. Let's take them in order.

I ran everything twice, against `strands-decider` and against TypeSafe's Jev through
OpenRouter. Both take the same state plus typed questions shape, so the integration has one
interface and two clients behind it. Partly that is because I wanted a second opinion from a
model I had no hand in, given I am hardly impartial about the first one, and partly because
if the gating idea only worked on our model it would be a much less interesting idea.

One structural note before the questions. This is the first of these integrations where I
could not use a clean extension point. Writing about Jev in Strands last time, six of my
seven experiments needed no wrapper, no fork and no patch. This one needed a subclass of the
session manager, overriding `retrieve_customer_context` and nothing else. Seventy lines of
policy replaced, about 1,300 lines of plumbing inherited untouched, because the write path is
not the part I am questioning.

## Question one: is a lookup worth making?

This one is cheap to ask, because it runs before any network call and needs one `noul`
question. Here is the wording I reached for first:

> Answering this message well depends on recalling a stored personal fact or preference
> about this specific user.

That scores 65% on 20 labelled utterances, which is close to worthless when the baseline I
am trying to beat is "always retrieve". The errors are not even politely distributed. "What's the capital of France?" came back at 0.597 and "review this function
and match my usual style" at 0.316, which is the wrong way round.

The same proposition, with the boundary written into `criteria`, scores 100% on the same
set in the same minute, lowest positive 0.284 against highest negative 0.086.

![Two strip plots of the same 20 messages. In the top panel, using the obvious wording, the messages that need memory and the messages that do not are mixed together across the probability range, 65% accuracy. In the bottom panel, using the same question with explicit true and false criteria, the two groups separate completely with a clear gap, 100% accuracy.](https://blog.mikegchambers.com/assets/images/decider-gated-memory/gate-wording.jpg)

| gate wording | best accuracy | margin |
| --- | --- | --- |
| the plain version above | 65% | -0.281 |
| the same question, with `criteria` | 100% | +0.198 |
| "would a profile lookup change the answer" | 90% | -0.025 |
| inverted, "answerable without knowing the user" | 95% | -0.118 |
| a four level `score` version | 95% | -0.049 |

I am slightly embarrassed to report this, because the last time I wrote about one of these
models I concluded that question design was most of the work, and then I went and wrote a
lazy question. The `criteria` field is where nearly all of the quality lives. If you take
one thing from this post, take that, and take that it costs 20 requests to find out, since
all five candidate wordings go in a single call per utterance.

## Question two: does this record belong in the prompt?

Harder, and this is where I got it wrong in a way that is worth describing.

My first attempt used the obvious wording with an absolute cut at 0.5. It dropped "the
user's go-to drink is a decaf oat flat white", at 0.484, in reply to "I'd like a drink, what
should I get?". It also dropped "vegetarian diet" at 0.499 for "what should I have for
dinner?". The single most useful record, binned, with the gate working perfectly upstream of
it.

What I had done was measure one thing when there were two. So I started scoring both,
across four queries with about nine records each:

- **Separation.** Pool every judgement. Is there one absolute threshold that splits
  relevant from irrelevant?
- **Ranking.** Within a single query, do the relevant records outrank the irrelevant ones?
  Per query AUC, and precision at k where k is the number of truly relevant records.

| validator wording | pooled margin | best acc | mean AUC | mean P@k |
| --- | --- | --- | --- | --- |
| plain | -0.022 | 80% | 0.835 | 69% |
| with `criteria` | +0.038 | 100% | 1.000 | 100% |
| topical overlap | -0.401 | 77% | 1.000 | 100% |
| counterfactual, "would omitting it hurt" | -0.137 | 74% | 0.490 | 29% |
| graded `score` | -0.105 | 91% | 0.935 | 85% |

Look at the topical overlap row. Perfect ranking, AUC 1.000 and precision at k of 100%, and
useless at any absolute threshold, margin -0.401. Had I measured only separation I would
have discarded a wording that ranks flawlessly. Had I measured only ranking I would have
shipped it behind a fixed threshold and watched it drop everything. One number hides
whichever of the two you did not think to look at, and I only found this because the first
version failed loudly enough to make me look.

The counterfactual row is the one I keep thinking about. AUC 0.490 and precision at k of
29% put it slightly below chance, so it is not merely weak, it is pointed the wrong way.
"An assistant that did not know this would give a noticeably worse answer" reads to me like
a *sharper* question than "is this useful". My guess, and it is only a guess, is that it
asks for a counterfactual judgement of value rather than a property you can check against
the text in front of you. I would not want to defend that explanation without more cases.

The winning wording puts relevant records at 0.379 and up, irrelevant at 0.341 and down, so
I set the threshold at 0.36. That is a gap of 0.038, which is narrow, and I would retune
before trusting it anywhere else.

## One confound I nearly shipped

AgentCore's strategies do not return the same shape as each other. `SEMANTIC` gives prose.
`USER_PREFERENCE` gives JSON:

```json
{"context": "Ordering a drink or beverage",
 "preference": "Dislikes sparkling water; prefers still water only",
 "categories": ["beverages", "water"]}
```

The stock manager passes that through as-is, and a frontier model copes. The classifier
copes less evenly, and the bias was systematic rather than noisy. JSON records scored above
prose records more or less regardless of relevance. A threshold tuned on prose therefore let
JSON through, which is how "always books an aisle seat" survived a question about what to
drink.

Rendering both strategies into one prose style first ("When ordering a drink or beverage:
Dislikes sparkling water; prefers still water only") sorted it out. On the drink query the
kept set went from 10 records to 7, and the three that flipped were the indentation
preference, the favourite cuisine and the aisle seat.

I apply that rendering on both sides of the comparison below. Otherwise I would be measuring
my own tidying up.

## What it bought

Eight turns, same store, Claude Sonnet 4.5 answering. The baseline arm is the stock
behaviour, which I get by passing `decider=None` so every gate branch is bypassed rather
than by running different code.

| | stock | gated, 2b | gated, Jev |
| --- | --- | --- | --- |
| memory retrievals | 8 | 3 | 3 |
| records injected | 147 | 25 | 16 |
| context injected | ~2,471 tokens | ~369 tokens | ~253 tokens |
| memory service time | 4,230 ms | 1,900 ms | 1,745 ms |
| decider calls | 0 | 14 (4,867 ms) | 14 (7,210 ms) |

![Three paired bar charts comparing the stock session manager with the gated version over an eight turn conversation. Memory API calls fall from 8 to 3, a 62% drop. Records in the prompt fall from 147 to 25, an 83% drop. Tokens of memory context fall from 2,471 to 369, an 85% drop.](https://blog.mikegchambers.com/assets/images/decider-gated-memory/results.jpg)

The gate closed on the five turns you would hope for, "Hello!", "What is 17 times 23?",
"Thanks!", "Can you explain what a bloom filter is?" and "ok cool", and opened on the three
that needed memory. Answers held up. Both arms recommend the decaf oat flat white, both
remember the aisle seat and the home airport when a flight comes up, both suggest vegetarian
pho with a peanut warning. The gated arm does it on about 15% of the context.

## What this does not show

Wall clock was 29.9 seconds gated against 29.3 stock. That is a wash, and the honest reading
is that I have not demonstrated a latency win.

The reason is mundane, and it is my own fault rather than the model's. For this experiment I
was calling a copy of the model on a remote SageMaker endpoint, from a laptop in a different
region, so each call cost roughly 350ms of round trip against 50 to 70ms of actual work at
the far end. Fourteen calls of avoidable network is about the 2.3 seconds saved on the memory
service, which is why the two arms finish together.

That is a deployment choice I made early and should have revisited, because the released
model does not work that way. It is a `pip install` that runs on the machine you are already
on, at around 153ms on an M3 MacBook. A local call has no round trip to pay for, so I would
expect the latency column to go positive. I have not run that configuration, so treat it as
the obvious next measurement rather than a result.

The comparison is also unfair to the stock manager in one direction that I should name. It is
tuned for a general case and I tuned my thresholds on this store, with these 19 records, using
labels I wrote. A fixed `relevance_score` would also look much better on a store whose scores
were well separated. I have shown that *this* store defeats a fixed threshold, not that
fixed thresholds are a bad idea in general.

So the two claims I will stand behind are the 85% reduction in injected context and five of
eight memory calls avoided. Everything else here is direction.

## Failing open hides outages

Both gates fail open. If the classifier errors, the gate retrieves and the validator keeps
the batch, so a dead dependency costs you the optimisation rather than the agent's memory.
I am fairly confident that is the right default.

It also bit me immediately. Making the endpoint name configurable, I set the default to a
name that does not exist. Every decision errored, every one failed open, and the demo
produced entirely correct answers. The only evidence was the stats block reading oddly:

```
gate decisions             8  (open 0, closed 0)
memory retrievals made     8  (baseline would be 8, saved 0)
```

Eight decisions, none of which opened or closed anything. I had to read my own code to
work out why, which is not a great sign. There is now a counter that says so loudly:

```
!! decider errors          16  (failed open -- results below reflect UNGATED behaviour)
```

The general lesson, which I suspect generalises past this project, is that fail-open and
observable are not the same property, and a system that degrades silently to "the old
behaviour" is indistinguishable from a system that is working. Count the failures.

## Where that leaves me

The division of labour I end up with is roughly this. The embedding is good at "what in
this store is about this subject", and that is not the question the agent needs answered on
a given turn. "Is this worth saying right now" is a judgement against evidence already in
hand, which seems to be the shape these small classifiers are reliably decent at.

The thing I did not expect going in is that the gate is both the cheaper half and the
easier half. Deciding whether to search at all is a judgement about one short message, and
one carefully worded question got it to 100% on my set. Deciding whether each record
belongs took four wordings and is where all the threshold fragility sits. If you only build
one of the two, build the gate.

Three things I have not built and would try next. Run the classifier locally, the way it is
actually shipped, and settle the latency column properly. Route rather than only gate, since
the gate is a single `noul` and a `choice` in the same request could pick which namespace is
worth searching, so a question about drinks never searches travel memories. And see whether
the ranking signal can replace the absolute threshold, because ranking was stable for both
models at AUC 1.000 across every run while the thresholds were not. Jev's absolute values
drifted enough between runs to move its best threshold, though the 2b model's stayed put.

So, did we earn the word in the use case list? I think so, with a caveat I would not have
thought to write before doing this. The mechanism works, and on this store it removed five
of eight memory calls and 85% of the injected context without costing the agent anything it
needed. But "use a decision model for memory" is not the useful instruction. The useful
instruction is "write the question twice, measure both, and measure ranking separately from
separation", because the first wording I tried was worth 65% and would have made the whole
thing look like a dead end. The mechanism was never the hard part.

A closing warning on the code. It is a subclass reaching into a part of the
`bedrock-agentcore` SDK that was refactored once while I was working on it. Treat it as a
spike rather than a library. It will break.
