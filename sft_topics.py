"""Topic-based SFT corpus generator — coding, physics, math."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable

from data_jsonl import accept_training_text, row_training_text

SFT_TOPICS: tuple[str, ...] = ("coding", "physics", "math")

TOPIC_LABELS: dict[str, str] = {
    "coding": "Coding & algorithms",
    "physics": "Physics & mechanics",
    "math": "Mathematics & proofs",
}


def topic_jsonl_filename(topic: str) -> str:
    if topic not in SFT_TOPICS:
        raise ValueError(f"Unknown SFT topic: {topic}")
    return f"sft-{topic}.jsonl"


def merged_topics_filename() -> str:
    return "sft-topics-merged.jsonl"


def _first_sentence(text: str) -> str:
    parts = text.split(". ")
    head = (parts[0] if parts else text).strip()
    if head and not head.endswith("."):
        head += "."
    return head


# Curated seeds — each expands to multiple user/assistant SFT rows.
_TOPIC_SEEDS: dict[str, list[dict[str, Any]]] = {
    "coding": [
        {
            "id": 101,
            "tier": 1,
            "token_class": "concept",
            "title": "Binary search invariant",
            "text": (
                "Binary search maintains the invariant that the answer lies in the half-open interval [lo, hi). "
                "Each step compares the midpoint value to the target and discards one half. "
                "Time complexity is O(log n) when random access is O(1). "
                "Common bugs: using inclusive bounds on both ends without updating correctly, "
                "or computing mid as (lo+hi)/2 instead of lo + (hi-lo)//2 which overflows in fixed-width integers."
            ),
        },
        {
            "id": 102,
            "tier": 1,
            "token_class": "example",
            "title": "Two-pointer merge",
            "text": (
                "To merge two sorted arrays in linear time, keep indices i and j at the starts. "
                "Compare A[i] and B[j], append the smaller to the output, and advance that index. "
                "When one array is exhausted, append the remainder of the other. "
                "This pattern generalizes to removing duplicates, finding pairs with a target sum, and "
                "partitioning linked lists without extra memory beyond output."
            ),
        },
        {
            "id": 103,
            "tier": 2,
            "token_class": "technical_note",
            "title": "Hash map amortized O(1)",
            "text": (
                "Average-case O(1) lookup in hash tables assumes a good hash function and load factor control. "
                "Python dicts resize when roughly two-thirds full. "
                "For counting frequencies, defaultdict(int) or Counter avoids KeyError boilerplate. "
                "For two-sum, store value→index while scanning; if target - current exists in the map, return the pair."
            ),
        },
        {
            "id": 104,
            "tier": 2,
            "token_class": "concept",
            "title": "Dynamic programming",
            "text": (
                "DP applies when optimal substructure and overlapping subproblems exist. "
                "Define state (often dp[i] or dp[i][j]), recurrence, base cases, and iteration order. "
                "Bottom-up tabulation avoids recursion depth; top-down memoization is faster to prototype. "
                "Space can often be compressed to O(1) or O(n) when only recent rows matter, as in Fibonacci or LIS variants."
            ),
        },
        {
            "id": 105,
            "tier": 2,
            "token_class": "example",
            "title": "BFS shortest path",
            "text": (
                "Breadth-first search on unweighted graphs finds shortest paths in edge count. "
                "Use a queue, mark visited when enqueuing (not when dequeuing) to avoid duplicates. "
                "For grids, encode cells as (r,c) tuples; 4-neighbor or 8-neighbor depends on problem. "
                "Track parent pointers or distance array if you must reconstruct the path."
            ),
        },
        {
            "id": 106,
            "tier": 3,
            "token_class": "technical_note",
            "title": "Testing edge cases",
            "text": (
                "Always test empty input, single element, duplicates, already-sorted and reverse-sorted data, "
                "and maximum constraint sizes. For strings: empty, single char, all same char, Unicode if relevant. "
                "Property-based tests (Hypothesis) catch off-by-one errors that fixed examples miss."
            ),
        },
    ],
    "physics": [
        {
            "id": 201,
            "tier": 1,
            "token_class": "definition",
            "title": "Newton's second law",
            "text": (
                "Newton's second law states that the net force on a particle equals the time derivative of momentum: "
                "F_net = dp/dt. For constant mass, F = ma. "
                "Forces are vectors; sum components independently. "
                "Free-body diagrams isolate the object and show all external forces before applying ΣF = ma."
            ),
        },
        {
            "id": 202,
            "tier": 1,
            "token_class": "concept",
            "title": "Conservation of energy",
            "text": (
                "In an isolated system, total energy (kinetic + potential + internal) is conserved. "
                "Mechanical energy K + U is conserved only when non-conservative work (friction, air drag) is negligible. "
                "For a block sliding down a frictionless ramp: mgh = ½mv² at the bottom. "
                "Always define the zero of potential energy consistently."
            ),
        },
        {
            "id": 203,
            "tier": 2,
            "token_class": "theorem",
            "title": "Kinematics with constant acceleration",
            "text": (
                "With constant acceleration a, displacement is x = x₀ + v₀t + ½at² and velocity is v = v₀ + at. "
                "Eliminating time gives v² = v₀² + 2a(x − x₀). "
                "These scalar equations apply per axis when acceleration is constant along that axis. "
                "Sign conventions matter: choose positive direction and stick to it."
            ),
        },
        {
            "id": 204,
            "tier": 2,
            "token_class": "concept",
            "title": "Rotational dynamics",
            "text": (
                "Torque τ = r × F causes angular acceleration: τ = Iα for a rigid body about a fixed axis. "
                "Moment of inertia I depends on mass distribution; parallel-axis theorem: I = I_cm + Md². "
                "Rolling without slipping relates v = ωR and energy splits between translation and rotation: "
                "K_total = ½mv² + ½Iω²."
            ),
        },
        {
            "id": 205,
            "tier": 2,
            "token_class": "technical_note",
            "title": "Units and dimensional analysis",
            "text": (
                "SI base units: kg, m, s, A, K, mol, cd. "
                "Dimensional analysis checks equations: both sides must have the same dimensions. "
                "Deriving a formula up to a dimensionless constant is often possible from powers of variables alone. "
                "Never add quantities with incompatible dimensions."
            ),
        },
        {
            "id": 206,
            "tier": 3,
            "token_class": "concept",
            "title": "Electric field and potential",
            "text": (
                "The electric field E is force per unit charge: F = qE. "
                "For a point charge, E = kq/r² radially. "
                "Potential V relates to E by E = −∇V; for point charges, V = kq/r. "
                "Equipotential surfaces are perpendicular to field lines. "
                "Capacitance C = Q/V; energy stored U = ½CV² = ½QV."
            ),
        },
    ],
    "math": [
        {
            "id": 301,
            "tier": 1,
            "token_class": "definition",
            "title": "Mathematical induction",
            "text": (
                "Proof by induction: (1) base case P(0) or P(1); (2) inductive step assuming P(k) prove P(k+1). "
                "Strong induction assumes P(0)…P(k) to prove P(k+1). "
                "Works for statements indexed by natural numbers, especially sums, divisibility, and recursive definitions."
            ),
        },
        {
            "id": 302,
            "tier": 1,
            "token_class": "theorem",
            "title": "Pythagorean theorem",
            "text": (
                "In a right triangle with legs a, b and hypotenuse c, a² + b² = c². "
                "Proofs include similar-triangle dissection and area rearrangement. "
                "Generalizes to Euclidean distance in ℝⁿ. "
                "Converse: if a² + b² = c², the triangle is right-angled opposite c."
            ),
        },
        {
            "id": 303,
            "tier": 2,
            "token_class": "concept",
            "title": "Limits and continuity",
            "text": (
                "lim_{x→a} f(x) = L means values of f(x) approach L as x approaches a (ε–δ formalization). "
                "f is continuous at a if lim_{x→a} f(x) = f(a). "
                "Polynomials, exponentials, and trig functions are continuous on their domains. "
                "Intermediate Value Theorem: continuous f on [a,b] hits every value between f(a) and f(b)."
            ),
        },
        {
            "id": 304,
            "tier": 2,
            "token_class": "theorem",
            "title": "Fundamental theorem of calculus",
            "text": (
                "If F′(x) = f(x) on [a,b], then ∫_a^b f(x)dx = F(b) − F(a). "
                "Part 1: d/dx ∫_a^x f(t)dt = f(x) for continuous f. "
                "Connects differentiation and integration as inverse operations. "
                "Substitution and integration by parts follow from the chain and product rules."
            ),
        },
        {
            "id": 305,
            "tier": 2,
            "token_class": "concept",
            "title": "Linear algebra basics",
            "text": (
                "A matrix represents a linear map. "
                "Ax = b is solvable uniquely when A is invertible (det ≠ 0 for square A). "
                "Eigenvalues λ satisfy Av = λv; diagonalization simplifies powers of A. "
                "Dot product u·v = ‖u‖‖v‖cos θ; orthogonal vectors have zero dot product."
            ),
        },
        {
            "id": 306,
            "tier": 3,
            "token_class": "technical_note",
            "title": "Combinatorics counting",
            "text": (
                "Addition rule: disjoint cases add. Multiplication rule: independent stages multiply. "
                "Permutations of n distinct objects: n!. "
                "Combinations n choose k = n!/(k!(n−k)!). "
                "Inclusion–exclusion counts unions: |A∪B| = |A| + |B| − |A∩B|. "
                "Pigeonhole principle: n+1 items into n boxes forces a box with at least two."
            ),
        },
    ],
}


def _prompts_for_seed(topic: str, seed: dict[str, Any]) -> list[str]:
    title = str(seed.get("title") or "this topic")
    token_class = str(seed.get("token_class") or "concept").replace("_", " ")
    tier = seed.get("tier", 1)
    text = str(seed.get("text") or "")
    first = _first_sentence(text)
    prompts = [
        f"Explain {title} clearly.",
        f"What should I know about {title}?",
        f"Teach me {title} (tier {tier}).",
        f"Give an intuitive explanation of {title}.",
    ]
    if token_class in ("theorem", "definition", "axiom"):
        prompts.append(f"State and explain the {token_class} for {title}.")
    if topic == "coding":
        prompts.extend(
            [
                f"How would you implement {title} in Python?",
                f"What are common bugs when applying {title}?",
                f"Walk through an example using {title}.",
            ]
        )
    elif topic == "physics":
        prompts.extend(
            [
                f"How do I set up a problem involving {title}?",
                f"What are the units and sign conventions for {title}?",
                f"Give a worked example for {title}.",
            ]
        )
    elif topic == "math":
        prompts.extend(
            [
                f"Prove or justify the key step in {title}.",
                f"Show a step-by-step example for {title}.",
                f"What are prerequisites for understanding {title}?",
            ]
        )
    if len(first) > 40:
        prompts.append(f"Elaborate: {first}")
    return prompts


def seed_to_sft_rows(topic: str, seed: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(seed.get("text") or "").strip()
    if len(text) < 80:
        return []
    seed_id = seed.get("id", "?")
    title = str(seed.get("title") or topic)
    token_class = str(seed.get("token_class") or "concept")
    principle = f"{TOPIC_LABELS.get(topic, topic)} — {title} ({token_class})"
    assistant = text[:6000]
    source = f"sft-topic://{topic}/{seed_id}"
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for user in _prompts_for_seed(topic, seed):
        user = user.strip()
        if not user or user in seen:
            continue
        seen.add(user)
        out.append(
            {
                "principle": principle,
                "user": user,
                "assistant": assistant,
                "source": source,
                "topic": topic,
                "seed_id": seed_id,
                "tier": seed.get("tier"),
                "token_class": token_class,
            }
        )
    return out


def expand_topic_seeds(topic: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in _TOPIC_SEEDS.get(topic, []):
        rows.extend(seed_to_sft_rows(topic, seed))
    return rows


def _row_key(row: dict[str, Any]) -> str:
    text = row_training_text(row)
    if text:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def _dedupe_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for row in candidates:
        text = row_training_text(row)
        if not accept_training_text(text):
            continue
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def _nemotron_to_sft(row: dict[str, Any]) -> dict[str, Any] | None:
    problem = row.get("problem")
    if not isinstance(problem, str) or len(problem.strip()) < 40:
        return None
    assistant_parts: list[str] = []
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").lower()
            content = msg.get("content") or msg.get("text")
            if isinstance(content, str) and content.strip() and role in ("assistant", "user"):
                if role == "assistant":
                    assistant_parts.append(content.strip())
    if not assistant_parts:
        expected = row.get("expected_answer")
        if isinstance(expected, str) and len(expected.strip()) >= 40:
            assistant_parts.append(expected.strip())
    if not assistant_parts:
        return None
    return {
        "principle": "Mathematics — problem solving",
        "user": problem.strip()[:2000],
        "assistant": "\n\n".join(assistant_parts)[:6000],
        "source": "hf://nemotron-math",
        "topic": "math",
    }


def _unsolved_to_sft(row: dict[str, Any]) -> dict[str, Any] | None:
    statement = row.get("statement")
    if not isinstance(statement, str) or len(statement.strip()) < 40:
        return None
    background = row.get("background")
    assistant = statement.strip()
    if isinstance(background, str) and len(background.strip()) > 80:
        assistant = f"{background.strip()}\n\nProblem:\n{statement.strip()}"
    return {
        "principle": "Mathematics — open problem context",
        "user": f"Explain the following problem and approach strategies:\n{statement.strip()[:1500]}",
        "assistant": assistant[:6000],
        "source": "hf://unsolved-math",
        "topic": "math",
    }


def _trace_to_sft(row: dict[str, Any], *, topic: str, source: str, principle: str) -> dict[str, Any] | None:
    prompt = row.get("prompt") or row.get("input") or row.get("question")
    completion = row.get("completion") or row.get("output") or row.get("answer")
    trace = row.get("trace") or row.get("cot")
    if isinstance(prompt, str) and isinstance(completion, str) and len(prompt.strip()) >= 20 and len(completion.strip()) >= 80:
        assistant = completion.strip()
        if isinstance(trace, str) and len(trace.strip()) > 40:
            assistant = f"{trace.strip()}\n\n{assistant}"
        return {
            "principle": principle,
            "user": prompt.strip()[:2000],
            "assistant": assistant[:6000],
            "source": source,
            "topic": topic,
        }
    text = row.get("text")
    if isinstance(text, str) and len(text.strip()) >= 120:
        parts = text.strip().split("\n\n", 1)
        if len(parts) == 2:
            return {
                "principle": principle,
                "user": parts[0][:2000],
                "assistant": parts[1][:6000],
                "source": source,
                "topic": topic,
            }
    return None


def _hf_augment_rows(topic: str, *, limit: int, hf_token: str | None) -> list[dict[str, Any]]:
    if not hf_token or limit <= 0:
        return []
    rows: list[dict[str, Any]] = []
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    factories: list[tuple[str, Callable[[], Any], Callable[[dict[str, Any]], dict[str, Any] | None]]] = []
    if topic == "coding":
        sol_url = os.environ.get(
            "SOL_TRACES_URL",
            "https://huggingface.co/datasets/greghavens/gpt-5.6-sol-coding-and-debugging-traces/resolve/main/data/train.jsonl",
        )
        factories.append(
            (
                "sol",
                lambda: load_dataset("json", data_files=sol_url, split="train", streaming=True, token=hf_token),
                lambda r: _trace_to_sft(
                    r,
                    topic="coding",
                    source="hf://sol-traces",
                    principle="Coding — trace-driven debugging",
                ),
            )
        )
    elif topic == "math":
        factories.append(
            (
                "nemotron",
                lambda: load_dataset(
                    "nvidia/Nemotron-SFT-Math-v4",
                    split="train",
                    streaming=True,
                    token=hf_token,
                ),
                _nemotron_to_sft,
            )
        )
        unsolved_url = os.environ.get(
            "UNSOLVED_MATH_URL",
            "https://huggingface.co/datasets/ulamai/UnsolvedMath/resolve/main/problems.json",
        )
        factories.append(
            (
                "unsolved",
                lambda: load_dataset("json", data_files=unsolved_url, split="train", streaming=True, token=hf_token),
                _unsolved_to_sft,
            )
        )
    # physics: seeds only for now

    per_source = max(1, limit // max(1, len(factories)))
    for _name, factory, convert in factories:
        count = 0
        try:
            stream = factory()
            for row in stream:
                if not isinstance(row, dict):
                    continue
                sft = convert(row)
                if sft:
                    rows.append(sft)
                    count += 1
                if count >= per_source or len(rows) >= limit:
                    break
        except Exception:
            continue
        if len(rows) >= limit:
            break
    return rows[:limit]


def generate_topic_sft(
    topic: str,
    dest_dir: str,
    *,
    hf_augment: bool = False,
    hf_limit: int = 40,
    hf_token: str | None = None,
) -> dict[str, Any]:
    if topic not in SFT_TOPICS:
        raise ValueError(f"Unknown topic: {topic}")
    candidates = expand_topic_seeds(topic)
    hf_rows = 0
    if hf_augment:
        extra = _hf_augment_rows(topic, limit=hf_limit, hf_token=hf_token)
        candidates.extend(extra)
        hf_rows = len(extra)
    kept = _dedupe_rows(candidates)
    dest = os.path.join(dest_dir, topic_jsonl_filename(topic))
    os.makedirs(dest_dir, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    return {
        "topic": topic,
        "file": os.path.basename(dest),
        "path": dest,
        "seed_rows": len(expand_topic_seeds(topic)),
        "hf_rows": hf_rows,
        "input": len(candidates),
        "kept": len(kept),
        "skipped": len(candidates) - len(kept),
        "size_mb": round(size_mb, 3),
    }


def generate_topics_sft(
    topics: list[str] | None = None,
    dest_dir: str = "data",
    *,
    hf_augment: bool = False,
    hf_limit: int = 40,
    hf_token: str | None = None,
    write_merged: bool = True,
) -> dict[str, Any]:
    chosen = [t for t in (topics or list(SFT_TOPICS)) if t in SFT_TOPICS]
    if not chosen:
        raise ValueError("No valid topics selected")
    per_topic: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for topic in chosen:
        stats = generate_topic_sft(
            topic,
            dest_dir,
            hf_augment=hf_augment,
            hf_limit=hf_limit,
            hf_token=hf_token,
        )
        per_topic.append(stats)
        path = stats["path"]
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    all_rows.append(json.loads(line))
    merged_stats: dict[str, Any] | None = None
    if write_merged and len(chosen) > 1:
        merged = _dedupe_rows(all_rows)
        merged_path = os.path.join(dest_dir, merged_topics_filename())
        with open(merged_path, "w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        merged_stats = {
            "file": merged_topics_filename(),
            "kept": len(merged),
            "size_mb": round(os.path.getsize(merged_path) / (1024 * 1024), 3),
        }
    return {
        "ok": True,
        "topics": per_topic,
        "merged": merged_stats,
        "total_kept": sum(t["kept"] for t in per_topic),
    }


def topic_catalog(dest_dir: str) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for topic in SFT_TOPICS:
        fname = topic_jsonl_filename(topic)
        path = os.path.join(dest_dir, fname)
        entry: dict[str, Any] = {
            "id": topic,
            "label": TOPIC_LABELS[topic],
            "file": fname,
            "data_source": f"sft-{topic}",
            "seed_count": len(_TOPIC_SEEDS.get(topic, [])),
            "exists": os.path.isfile(path),
            "rows": 0,
            "size_mb": 0.0,
        }
        if entry["exists"]:
            entry["size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 3)
            with open(path, encoding="utf-8") as handle:
                entry["rows"] = sum(1 for line in handle if line.strip())
        catalog.append(entry)
    merged_path = os.path.join(dest_dir, merged_topics_filename())
    if os.path.isfile(merged_path):
        catalog.append(
            {
                "id": "merged",
                "label": "All topics merged",
                "file": merged_topics_filename(),
                "data_source": None,
                "exists": True,
                "size_mb": round(os.path.getsize(merged_path) / (1024 * 1024), 3),
                "rows": sum(1 for line in open(merged_path, encoding="utf-8") if line.strip()),
            }
        )
    return catalog
