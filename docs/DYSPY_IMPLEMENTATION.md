# DSPy Developer Guidelines (Sept 2025) — Senior Analytics Engineer Edition

Audience: Senior analytics engineers (BigQuery/Snowflake/Vertex AI/Gemini users) building reliable, maintainable, *self‑improving* LLM systems.

Goal: Give you a practical, end‑to‑end playbook for programming with **DSPy** (Declarative Self‑improving Python): how to wire models (incl. Google GenAI SDK/Vertex AI), choose modules, compose programs, optimize prompts/weights, evaluate, observe, and ship to production.

---

## 1) The 30‑second mental model

* **Signatures → Modules → Programs → Optimizers**

  * **Signature** = typed contract (inputs/outputs).
  * **Module** = implementation strategy (e.g., `Predict`, `ChainOfThought`, `ReAct`, `ProgramOfThought`).
  * **Program** = composition of modules into a pipeline/agent.
  * **Optimizer** = an algorithm that *compiles* your program into better instructions/few‑shots/weights against a metric (e.g., Exact‑Match, Semantic F1).
* You write normal Python; DSPy handles prompt construction, few‑shot selection, tool use, caching, async/streaming, and optimization.

---

## 2) Install & project setup

### 2.1. Minimal install

```bash
# Astral UV (recommended)
uv venv
uv pip install -U dspy-ai google-genai mlflow

# or plain pip
pip install -U dspy-ai google-genai mlflow
```

### 2.2. Environment & keys

* **Google GenAI SDK (Gemini API)**: set `GEMINI_API_KEY`.
* **Vertex AI**: authenticate with ADC (service account key or `gcloud auth application-default login`) **or** use the GenAI SDK’s Vertex mode with project/location.
* Recommended: keep secrets in `.env` and load with `python-dotenv`.

### 2.3. Repo hygiene

* `/dspy/` module with programs, `/data/` for labeled examples, `/ops/` for MLflow run configs, `/scripts/` for RAG indexing, `/tests/` for program tests.

---

## 3) Configure LMs (incl. Google GenAI & Vertex)

DSPy speaks to many providers through a **unified LM interface**. Think *LiteLLM‑style* provider/model strings and pass‑through kwargs.

```python
import os, dspy

# Option A — Google Gemini via Google GenAI SDK (Developer API)
lm = dspy.LM(
    "gemini/gemini-1.5-pro",  # or "gemini/gemini-2.0-flash"
    api_key=os.environ["GEMINI_API_KEY"],
    model_type="chat",           # DSPy forwards provider-specific kwargs
    temperature=0.2,
)
dspy.configure(lm=lm)

# Option B — Vertex AI (service-backed Gemini) via provider string
lm = dspy.LM(
    "vertex_ai/gemini-1.5-pro",
    project="my-gcp-project",
    location="us-central1",
    temperature=0.2,
)
dspy.configure(lm=lm)

# Tip: You can reconfigure per test or per optimizer run.
```

**Notes**

* The provider name (e.g., `gemini`, `vertex_ai`) and kwargs are forwarded; check your account, quota, and model names.
* For local/proxy deployments that expose OpenAI‑compatible endpoints, use `openai/<model>` with `api_base` and `api_key`.

---

## 4) Core primitives: Signatures & Modules

### 4.1. Signatures (specs)

```python
import dspy

class TicketTriage(dspy.Signature):
    """Map a support ticket to {category, priority, owner}."""
    ticket: str = dspy.InputField()
    category: str = dspy.OutputField(desc="short label from {billing,bug,howto,other}")
    priority: str = dspy.OutputField(desc="P1..P4")
    owner: str = dspy.OutputField(desc="team or person")
```

### 4.2. Built‑in modules you’ll actually use

* **`Predict`** – baseline single‑shot predictor.
* **`ChainOfThought`** – adds explicit reasoning traces before final outputs.
* **`ProgramOfThought`** – lets the LLM write & run code (great for analytics/math).
* **`ReAct`** – tool‑using agent loop (search, Python tools, DB calls, RAG).
* **`MultiChainComparison`** – generate/compare multiple CoTs and pick/compose the best.
* **`BestOfN`**, **`Parallel`**, **`Refine`**, **`CodeAct`** – ensembling, parallel calls, post‑edits, code‑centric actions.

**Hello‑world**

```python
triage = dspy.Predict(TicketTriage)
print(triage(ticket="Checkout fails with card declined on Visa."))
```

**Reasoning upgrade**

```python
triage_reasoned = dspy.ChainOfThought(TicketTriage)
triage_reasoned(ticket="Checkout fails with card declined on Visa.")
```

---

## 5) Compose programs (pipelines & agents)

### 5.1. Pipeline composition

```python
class ClassifyThenExplain(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("issue -> category")
        self.explain  = dspy.ChainOfThought("issue, category -> rationale, action")

    def forward(self, issue: str):
        cat = self.classify(issue=issue).category
        out = self.explain(issue=issue, category=cat)
        return dict(category=cat, **out)
```

### 5.2. ReAct with tools (incl. ColBERTv2 retriever & Python)

```python
from typing import List

# Simple retriever tool (HTTP ColBERTv2 server or hosted index)
search = dspy.ColBERTv2(url="http://localhost:2017/wiki")

def topk(query: str, k: int = 5) -> List[str]:
    return [d["text"] for d in search(query, k=k)]

# Optional: a sandboxed Python tool for light data ops
py = dspy.PythonInterpreter()

agent = dspy.ReAct(
    "question -> answer",
    tools=[topk, py],
)
```

---

## 6) Optimizers: compile programs into better prompts/weights

Optimizers (a.k.a. *teleprompters*) search over instructions, demos, and even LM weights to improve your metrics.

### 6.1. Quick glossary

* **Few‑shot learners**: `LabeledFewShot`, `BootstrapFewShot`, `BootstrapFewShotWithRandomSearch`, `KNNFewShot`.
* **Instruction optimizers**: `MIPROv2`, `COPRO`, `GEPA`, `SIMBA`.
* **Finetuning**: `BootstrapFinetune` (distill to small models).
* **Program transforms**: `Ensemble`, `BetterTogether`.

### 6.2. Typical usage pattern

```python
from dspy.teleprompt import MIPROv2
from dspy.evaluate import answer_exact_match

# 1) Metric — define how to score predictions
metric = answer_exact_match

# 2) Tiny trainset — inputs (and optionally labels)
trainset = [
  dspy.Example(ticket="Visa card declined", category="billing", priority="P2", owner="payments").with_inputs("ticket"),
  dspy.Example(ticket="App crashes on login", category="bug", priority="P1", owner="auth").with_inputs("ticket"),
]

# 3) Program to optimize
program = dspy.ChainOfThought(TicketTriage)

# 4) Compile with MIPROv2 (light/medium/heavy search)
opt = MIPROv2(metric=metric, auto="light", num_threads=8)
program_optimized = opt.compile(program, trainset=trainset,
                                max_bootstrapped_demos=2, max_labeled_demos=2)
```

> Rule of thumb
>
> * **Few examples (<20)** → start with `BootstrapFewShot` or `MIPROv2(auto="light")`.
> * **Dozens–hundreds** → `MIPROv2` (medium/heavy). Consider `KNNFewShot` for data‑aware demos.
> * **Latency/cost critical** → `BootstrapFinetune` a small model; use `Ensemble` for reliability where cost allows.

---

## 7) Data handling, metrics & evaluation

### 7.1. Examples & splits

```python
# Dev/test split — you usually need far fewer labels than in classic ML
trainset = [ex1, ex2, ...]
devset   = [ex_eval1, ex_eval2, ...]

from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import SemanticF1, answer_exact_match

eval = Evaluate(
    devset=devset,
    metrics=[SemanticF1(), answer_exact_match],
    provide_traceback=True,
)
score = eval(program_optimized)
```

### 7.2. Common metrics

* `answer_exact_match` — strict string match.
* `SemanticF1()` — similarity with semantic tolerance (great for long‑form/Q\&A/RAG).
* `answer_passage_match` — groundedness to retrieved context.

### 7.3. Contract tests (fast unit tests)

* Freeze LM with `temperature=0` for determinism; assert type/schema.
* Use small golden sets per module; larger eval per program.

---

## 8) Observability, caching, streaming & async

### 8.1. Caching

* DSPy caches LM calls keyed by signature + inputs. Bypass a cache entry by setting a new `rollout_id` in `config`.

```python
pred = dspy.Predict("q -> a")
pred(q="1+1", config={"rollout_id": 1, "temperature": 1.0})
```

### 8.2. Inspect & debug

```python
from dspy.utils import inspect_history
inspect_history(program_optimized)  # view traces for each module call
```

### 8.3. Streaming tokens

```python
async for event in dspy.streamify(agent, field="answer")(question="Why is my card declined?"):
    # event.chunk, event.done, event.elapsed_ms
    ...
```

### 8.4. Async execution

* Most modules support `await module.acall(**inputs)`; implement `async def forward(...)` in custom modules.

---

## 9) MLflow autologging (experiments & lineage)

```python
import mlflow
from mlflow.dspy import autolog

mlflow.set_experiment("dspy-triage")
autolog(log_traces=True)

with mlflow.start_run(run_name="mipro_light_triage"):
    compiled = opt.compile(program, trainset=trainset)
    score = eval(compiled)
    mlflow.log_metric("semantic_f1", score["SemanticF1"])  # example
```

* Captures optimizer trials, prompts, demos, traces, and final compiled program.
* Treat every optimizer run like an experiment; store artifacts (compiled program state) for promotion.

---

## 10) Saving/loading compiled programs

```python
# Save only the state (instructions, demos, config)
program_optimized.save("artifacts/triage_state.json", save_program=False)

# Load later
restored = dspy.load("artifacts/triage_state.json")
```

* Promote by environment: **dev → staging → prod** with frozen state; keep eval dashboards per env.

---

## 11) RAG patterns that work in practice

### 11.1. Simple RAG (search → reason)

```python
class SimpleRAG(dspy.Module):
    def __init__(self, retriever, k=5):
        self.retriever = retriever
        self.k = k
        self.answer = dspy.ChainOfThought("context, question -> response")

    def forward(self, question: str):
        docs = [x["text"] for x in self.retriever(question, k=self.k)]
        return self.answer(context="\n\n".join(docs), question=question)
```

### 11.2. Reliability upgrades

* Swap `ChainOfThought` → `MultiChainComparison`.
* Add re‑ranking; keep `answer_passage_match` ≥ threshold.
* Optimize with `MIPROv2(auto="medium")` using `SemanticF1()`.

### 11.3. Indexing notes

* ColBERTv2 server returns JSON; set `simplify=True` if you just want texts.
* Keep an offline corpus snapshot per release for reproducibility.

---

## 12) Agentic workflows (ReAct) for data ops

Common tools:

* **Retrievers** (ColBERTv2, vector DBs), **HTTP**/**SQL** readers, **PythonInterpreter** (small computations), custom **BigQuery/Dataproc** functions.

Safety & control:

* Add a `Tool` that enforces schemas and rate limits.
* Gate finishing with checks (e.g., schema validation); log the final thought/action/observation triplets.

---

## 13) Adapters & strict JSON IO

When you need strict schemas (e.g., emitting JSON to a downstream job):

* **`JSONAdapter`**: enforce response schema.
* **`ChatAdapter`**: multi‑turn chat turns with memory.
* **`TwoStepAdapter`**: draft → verify/refine.

```python
from dspy.adapters import JSONAdapter

sig = dspy.Signature.from_str("text -> {title, tags: list[str]}")
extract = dspy.Predict(sig)
extract = JSONAdapter(extract, schema={"title": str, "tags": [str]})
```

---

## 14) Cost, latency & quality playbook

* **Start tiny**: 10–20 examples; `MIPROv2(auto="light")` or `BootstrapFewShot`.
* **Escalate**: add KNN‑assisted demos; raise search budget only when metric plateaus.
* **Latency controls**: prefer `Predict` over `CoT` on easy hops; parallelize with `Parallel` or batch inputs.
* **Cost controls**: cache aggressively, stream partials to UX, use smaller models for bootstrap; distill with `BootstrapFinetune` if volume is high.

---

## 15) Production checklist

* [ ] Program contracts captured as **Signatures**; golden tests for each module.
* [ ] **Optimizer** configs & budgets versioned; MLflow runs captured (trials, prompts, artifacts).
* [ ] **State** saved for every promoted build; reproducible RAG snapshots.
* [ ] **Observability**: traces on; error budgets and retry policies set.
* [ ] **Security**: PII redaction, allow‑list tools, request/response size limits.
* [ ] **SLOs**: latency p95, answer quality metric, cost per 1k requests.

---

## 16) Patterns by use case (quick templates)

### 16.1. Information extraction (strict JSON)

* Module: `ChainOfThought` → `JSONAdapter` → `Refine` (optional)
* Metric: exact match (by field), partial credit by Semantic F1.
* Optimizer: `MIPROv2` with small bootstrap demos.

### 16.2. Classification & routing

* Module: `Predict` or `BestOfN` for confidence, optional `ProgramOfThought` for rule‑heavy.
* Metric: accuracy / F1.
* Optimizer: start with `BootstrapFewShot`; graduate to `MIPROv2`.

### 16.3. Multi‑hop RAG & agents

* Module: `ReAct` + retriever tools, then `MultiChainComparison`.
* Metric: `SemanticF1()` and `answer_passage_match`.
* Optimizer: `MIPROv2(auto=\"medium\")`; consider `BetterTogether`/`Ensemble` for reliability.

---

## 17) Custom modules (when you need bespoke logic)

```python
class StructuredValidator(dspy.Module):
    def __init__(self):
        self.gen = dspy.Predict("raw -> json")
    def forward(self, raw: str):
        out = self.gen(raw=raw)
        # add custom Python validation/repair
        return out
```

* Subclass `dspy.Module`; declare sub‑modules in `__init__`; implement `forward`/`aforward`.

---

## 18) Troubleshooting cheat‑sheet

* **Model/provider errors**: verify provider string (e.g., `gemini/gemini-1.5-pro` vs `vertex_ai/gemini-1.5-pro`), credentials, and region.
* **Caching confusion**: bump `rollout_id` in `config` to force a fresh call.
* **Inconsistent evals**: fix temperature, normalize whitespace, canonicalize JSON keys.
* **Slow ReAct**: reduce `k`, add stop conditions, or skip CoT extraction when you don’t need it.

---

## 19) Next steps for your stack

* Wrap your *Launchpad/Pipeline* datasets as retriever tools; gate answers on `answer_passage_match`.
* Create a small, curated devset from recent tickets/incidents; run `MIPROv2(auto="light")` weekly and track in MLflow.
* Promote compiled state only if eval metric improves *and* cost/latency stay within SLOs.

---

## 20) Appendix — quick API index you’ll reach for

* **Modules**: `Predict`, `ChainOfThought`, `ProgramOfThought`, `ReAct`, `MultiChainComparison`, `Parallel`, `BestOfN`, `Refine`, `CodeAct`
* **Optimizers**: `MIPROv2`, `BootstrapFewShot`, `BootstrapFewShotWithRandomSearch`, `KNNFewShot`, `COPRO`, `GEPA`, `SIMBA`, `BootstrapFinetune`, `Ensemble`, `BetterTogether`
* **Tools**: `ColBERTv2`, `Embeddings`, `PythonInterpreter`
* **Metrics/Eval**: `Evaluate`, `SemanticF1`, `answer_exact_match`, `answer_passage_match`
* **Utils**: `streamify`, `inspect_history`, `configure_cache`, `enable_logging`, `disable_logging`

---

> **Keep it pragmatic**: small curated data + modular programs + the right optimizer will reliably beat hand‑prompting. Treat prompt/weight search like normal ML experiments, and wire it to your analytics/observability stack from day one.
