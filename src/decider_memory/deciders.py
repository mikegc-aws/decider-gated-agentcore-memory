"""Decision-model clients.

Two backends, one interface. Both speak the same `state` + `questions` schema, so
the session manager never has to care which one it got:

    SageMakerDecider  strands-decider-2b on a SageMaker/Triton endpoint
    JevDecider        Jev (TypeSafe's decisions model) via OpenRouter

The interface is deliberately **synchronous**. The thing we are gating --
`retrieve_customer_context` -- is called from a sync Strands hook and already fans
out over a ThreadPoolExecutor, so an async client would buy nothing and cost a lot
of plumbing.

The one performance rule that matters: **ask every question you have in a single
call.** Extra questions about the same state are nearly free (for the SageMaker
deployment, 1 question is ~47 ms and 7 questions ~67 ms), whereas a second round
trip is another full network hop. The session manager relies on this -- it
validates N retrieved memories with one N-question call, not N calls.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# --------------------------------------------------------------------- answers


@dataclass
class ChoiceAnswer:
    """One label out of a set, plus how sure the model is."""

    label: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.label} ({self.confidence:.2f})"


@dataclass
class ScoreAnswer:
    """A position on an ordered scale.

    `value` is on the level-index scale (0 .. levels-1), *not* 0..1. Use
    `.fraction` if you want it normalised -- reading `value` as though it were
    already normalised is the easiest mistake to make here, and it fails silently
    because the number still looks plausible.
    """

    value: float
    levels: int
    confidence: float
    label: str = ""

    @property
    def fraction(self) -> float:
        return self.value / (self.levels - 1) if self.levels > 1 else 0.0

    def __str__(self) -> str:
        return f"{self.label} ({self.value:.2f}/{self.levels - 1})"


# `noul` answers come back as a bare float: P(the proposition holds).
Answer = float | ChoiceAnswer | ScoreAnswer


# ------------------------------------------------------------------- questions


def noul(instructions: str, criteria: dict[str, str] | None = None) -> dict:
    """Probability that a proposition is true. Answered as a plain float.

    `criteria` optionally sharpens the boundary: {"true": "...", "false": "..."}.
    """
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria:
        q["criteria"] = criteria
    return q


def choice(instructions: str, options: dict[str, str | None] | list[str]) -> dict:
    """Pick one label. 2..255 options.

    Descriptions are worth writing -- they are the only place the model learns what
    a label is supposed to mean, and they are free at inference time.
    """
    criteria = {o: None for o in options} if isinstance(options, list) else options
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, levels: list[str]) -> dict:
    """A position on an ordered scale. `levels` runs low to high, 10 at most."""
    return {"type": "score", "instructions": instructions, "criteria": levels}


# -------------------------------------------------------------------- protocol


class DeciderError(RuntimeError):
    """The backend answered, but with an error body. See `.kind`."""

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind


@runtime_checkable
class Decider(Protocol):
    """A decision model: ask typed questions about some state, get typed answers."""

    name: str

    def ask(self, state: Any, questions: dict[str, dict]) -> dict[str, Answer]:
        """Answer every question in `questions` about `state`, in one round trip."""
        ...


def _normalise(raw_answers: dict[str, dict]) -> dict[str, Answer]:
    """Both backends return the same answer shapes; flatten them identically."""
    out: dict[str, Answer] = {}
    for name, raw in raw_answers.items():
        kind = raw.get("type")
        if kind == "choice":
            out[name] = ChoiceAnswer(
                label=raw["choice"],
                confidence=raw.get("confidence", 0.0),
                probabilities=raw.get("probabilities", {}),
            )
        elif kind == "score":
            legend = raw.get("legend", {})
            value = raw["score"]
            out[name] = ScoreAnswer(
                value=value,
                levels=max(len(legend), 1),
                confidence=raw.get("confidence", 0.0),
                label=legend.get(str(int(round(value))), ""),
            )
        else:  # noul
            out[name] = raw["noul"]
    return out


# ------------------------------------------------------- SageMaker / Triton 2b


class SageMakerDecider:
    """strands-decider-2b on a SageMaker real-time endpoint (Triton backend).

    There is no plain URL to curl: every request is SigV4-signed, so this goes
    through boto3 with the *endpoint name*.

    The payload travels inside a Triton tensor envelope, and two details there are
    not negotiable:
      - `data[0]` is the decider request JSON-encoded as a *string*, not a nested
        object.
      - request `shape` is `[1, 1]`, not `[1]` -- the model declares
        max_batch_size > 0 so Triton prepends a batch dimension. (The *response*
        shape is `[1]`. The asymmetry is real.)
    """

    name = "strands-decider-2b"

    def __init__(
        self,
        endpoint: str | None = None,
        region: str | None = None,
        max_in_flight: int = 64,
    ) -> None:
        import boto3
        from botocore.config import Config

        # Point this at your own endpoint via DECIDER_ENDPOINT / DECIDER_REGION.
        #
        # Note this is NOT how strands-decider is meant to be run. The released
        # model is a local `pip install strands-decider` (~153 ms on an M3
        # MacBook). This class talks to a SageMaker/Triton endpoint because that
        # is how I had it deployed, and the round trip from outside the region is
        # the sole reason the PoC shows no latency win. See the README.
        self.endpoint = endpoint or os.environ.get("DECIDER_ENDPOINT", "strands-decider-g6")
        region = region or os.environ.get("DECIDER_REGION", "us-west-2")
        self._client = boto3.client(
            "sagemaker-runtime",
            region_name=region,
            config=Config(
                # botocore defaults to a pool of 10, which silently serialises
                # anything above 10 concurrent calls -- you end up measuring the
                # client rather than the model.
                max_pool_connections=max(10, max_in_flight),
                retries={"max_attempts": 2, "mode": "standard"},
                read_timeout=60,  # SageMaker's own hard ceiling for real-time
                connect_timeout=10,
            ),
        )

    def ask(self, state: Any, questions: dict[str, dict]) -> dict[str, Answer]:
        inner = json.dumps({"state": state, "questions": questions})
        envelope = {
            "inputs": [
                {
                    "name": "REQUEST_JSON",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [inner],
                }
            ]
        }
        resp = self._client.invoke_endpoint(
            EndpointName=self.endpoint,
            ContentType="application/json",
            Body=json.dumps(envelope),
        )
        out = json.loads(resp["Body"].read().decode("utf-8"))

        # A Triton-level failure (bad envelope) comes back with no `outputs`.
        if "error" in out and "outputs" not in out:
            raise DeciderError(str(out["error"]), "envelope")

        for tensor in out.get("outputs") or []:
            if tensor.get("name") == "RESPONSE_JSON":
                data = tensor.get("data") or []
                if not data:
                    raise DeciderError("RESPONSE_JSON carried no data", "internal")
                body = json.loads(data[0])
                # Caller errors are reported in the body at HTTP 200, deliberately,
                # so a malformed payload is distinguishable from the model being down.
                if "error" in body:
                    err = body["error"]
                    raise DeciderError(
                        err.get("message", "unknown"), err.get("type", "invalid_request")
                    )
                return _normalise(body.get("answers", {}))

        raise DeciderError(f"no RESPONSE_JSON in {sorted(out)}", "internal")


# ------------------------------------------------------------------------ Jev


class JevDecider:
    """Jev, TypeSafe's decisions model, via OpenRouter's /decisions endpoint.

    Same question schema as the SageMaker deployment, different transport and a
    plain bearer token. Useful as a second opinion and as a check that the gating
    idea is not an artefact of one particular model.
    """

    name = "jev"
    MODEL = "~typesafe/jev-latest"  # the leading ~ is part of the id
    ENDPOINT = "https://openrouter.ai/api/alpha/decisions"

    def __init__(self, timeout: float = 30.0, api_key: str | None = None) -> None:
        import httpx

        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("no API key: set OPENROUTER_API_KEY")
        self._client = httpx.Client(
            timeout=timeout, headers={"Authorization": f"Bearer {key.strip().strip(chr(34))}"}
        )
        self.cost = 0.0

    def close(self) -> None:
        self._client.close()

    def ask(self, state: Any, questions: dict[str, dict]) -> dict[str, Answer]:
        resp = self._client.post(
            self.ENDPOINT,
            json={"model": self.MODEL, "state": state, "questions": questions},
        )
        if resp.status_code != 200:
            raise DeciderError(f"HTTP {resp.status_code}: {resp.text[:300]}", "internal")
        body = resp.json()
        self.cost += body.get("usage", {}).get("cost", 0.0)
        return _normalise(body.get("answers", {}))


def make_decider(backend: str = "sagemaker", **kwargs: Any) -> Decider:
    """Build a decider by name: "sagemaker" (strands-decider-2b) or "jev"."""
    if backend in ("sagemaker", "decider", "strands-decider-2b"):
        return SageMakerDecider(**kwargs)
    if backend == "jev":
        return JevDecider(**kwargs)
    raise ValueError(f"unknown decider backend: {backend!r}")
