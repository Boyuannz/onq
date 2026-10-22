r"""Combine pinned benchmark texts and the evaluation AIME input for decontamination.

Public sources use pinned Hugging Face revisions or archives with SHA256.
The same prepared AIME 2024-2026 input used for evaluation is required locally.
Output rows are ``{benchmark, id, part, text}``, with empty texts dropped and
exact ``(benchmark, id, part, text)`` duplicates removed.

Lowercase labels are the Block-AP held-out prompts (``part=prompt``, ``id`` = row index
within the listed file group). Capitalized labels are the expanded evaluation sets:
multiple-choice sets emit ``core`` and ``core + "\n" + "\n".join(choices)``
(``question_choices``); HumanEval/MBPP additionally emit reference solutions.

    python -m OnQ preparedata benchmarks --aime90 AIME_DIR --out HELDOUT_QUERIES.jsonl

Note: ``lcb`` downloads the full LiveCodeBench release_v6 files (~4.5 GB, test cases inline)."""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
import re
import urllib.request
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]
GPQA_PWD = b"deserted-untie-orchid"  # published in the GPQA README (anti-crawl zip password)
# ---- PINS: name -> (repo or URL, revision or SHA256 of archive, [files]) ----------------
PINS = {
    "aime24": ("HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
               ["data/train-00000-of-00001.parquet"]),
    "aime25": ("opencompass/AIME2025", "a6ad95f611d72cf628a80b58bd0432ef6638f958",
               ["aime2025-I.jsonl", "aime2025-II.jsonl"]),
    "amc23": ("math-ai/amc23", "80815d37005feb82cd7f8fbc6901d5d3eff43057",
              ["test-00000-of-00001.parquet"]),
    "gpqa": ("https://github.com/idavidrein/gpqa/raw/56686c06f5e19865c153de0fdb11be3890014df7/dataset.zip",
             "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc",
             ["dataset/gpqa_diamond.csv"]),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
              ["main/test-00000-of-00001.parquet"]),
    "humaneval": ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                  ["openai_humaneval/test-00000-of-00001.parquet"]),
    "ifbench": ("allenai/IFBench_test", "2e8a48de45ff3bf41242f927254ca81b59ca3ae2",
                ["data/train-00000-of-00001.parquet"]),
    "ifeval": ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84",
               ["ifeval_input_data.jsonl"]),
    "lcb": ("livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
            ["test.jsonl"] + [f"test{k}.jsonl" for k in range(2, 7)]),  # release_v6 = test..test6
    "math500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", ["test.jsonl"]),
    "math_test": ("EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3",
                  [f"{s}/test-00000-of-00001.parquet" for s in MATH_SUBJECTS]),
    "mbpp": ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
             ["full/test-00000-of-00001.parquet", "full/validation-00000-of-00001.parquet"]),
    "mmlu": ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", ["all/test-00000-of-00001.parquet"]),
    "arc_c": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453",
              ["ARC-Challenge/test-00000-of-00001.parquet"]),
    "arc_e": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453",
              ["ARC-Easy/test-00000-of-00001.parquet"]),
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76",
                  ["data/validation-00000-of-00001.parquet"]),
    "mmlu_redux": ("edinburgh-dawg/mmlu-redux-2.0", "372ea425445d51e1ba1188c56e5e893f8138621f",
                   None),  # {subject}/data-00000-of-00001.arrow for the 57 MMLU subjects
    "openbookqa": ("allenai/openbookqa", "388097ea7776314e93a529163e0fea805b8a6454",
                   ["main/test-00000-of-00001.parquet"]),
    "piqa": ("https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip",
             "54d32a04f59a7e354396f321723c8d7ec35cc6b08506563d8d1ffcc15ce98ddd",
             ["physicaliqa-train-dev/dev.jsonl"]),
    "siqa": ("https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip",
             "ee073914a0fc33265cfbcfc50ec20df9b2e07809c3f420f599138d1f394ef5c3",
             ["socialiqa-train-dev/dev.jsonl"]),
    "truthfulqa": ("truthfulqa/truthful_qa", "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
                   ["multiple_choice/validation-00000-of-00001.parquet"]),
    "winogrande": ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
                   ["winogrande_xl/validation-00000-of-00001.parquet"]),
}

CACHE = None


def parse(name, data):
    if name.endswith(".parquet"):
        return pq.read_table(pa.BufferReader(data)).to_pylist()
    if name.endswith(".arrow"):
        return pa.ipc.open_stream(data).read_all().to_pylist()
    if name.endswith(".csv"):
        return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    return [json.loads(l) for l in data.decode("utf-8").splitlines() if l.strip()]


def fetch_url(url, sha):
    root = Path(CACHE or Path.home() / ".cache/huggingface") / "benchmarks_url"
    path = root / sha
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(url, timeout=300).read()
        path.with_suffix(".tmp").write_bytes(data)
        os.replace(path.with_suffix(".tmp"), path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"SHA256 mismatch for {url}")
    return zipfile.ZipFile(io.BytesIO(data))


def load(key, files=None, lines=None):
    """List of per-file row lists for pin ``key``; ``lines`` maps each JSONL row while streaming."""
    repo, rev, pinned = PINS[key]
    out = []
    for name in files or pinned:
        if repo.startswith("https://"):
            # The password only applies to the encrypted GPQA member; plain members ignore it.
            out.append(parse(name, fetch_url(repo, rev).read(name, pwd=GPQA_PWD)))
            continue
        path = hf_hub_download(repo, name, repo_type="dataset", revision=rev, cache_dir=CACHE)
        if lines:
            with open(path, encoding="utf-8") as f:
                out.append([lines(json.loads(l)) for l in f if l.strip()])
        else:
            out.append(parse(name, Path(path).read_bytes()))
    return out


def cat(files):
    return [r for rows in files for r in rows]


def prompts(files, field, per_file=False):
    """(id, 'prompt', text); ids restart per file when ``per_file`` (historical per-file loop)."""
    groups = files if per_file else [cat(files)]
    return [(i, "prompt", r[field]) for rows in groups for i, r in enumerate(rows)]


def mc(rows, core, opts):
    out = []
    for i, r in enumerate(rows):
        c = core(r)
        out += [(i, "core", c), (i, "question_choices", c + "\n" + "\n".join(map(str, opts(r))))]
    return out


def gpqa_questions(rows):
    """Order-preserving unique revised/pre-revision question texts (legacy gpqa_diamond_questions)."""
    texts = (r[k] for r in rows for k in ("Question", "Pre-Revision Question") if r[k].strip())
    return [{"question": t} for t in dict.fromkeys(texts)]


def mmlu_redux():
    # Historical order: 57 alphabetical subjects dealt round-robin into 7 evaluation shards.
    subjects = sorted({r["subject"] for r in load("mmlu")[0]})
    assert len(subjects) == 57
    order = [s for k in range(7) for s in subjects[k::7]]
    return cat(load("mmlu_redux", [f"{s}/data-00000-of-00001.arrow" for s in order]))


def build(aime_rows):
    """Yield pinned benchmark texts followed by the actual AIME evaluation questions."""
    gpqa = load("gpqa")[0]
    mmlu = load("mmlu")
    mbpp = load("mbpp")
    humaneval = load("humaneval")[0]
    choices = lambda r: r["choices"]["text"]
    blocks = [
        ("aime24", prompts(load("aime24"), "problem")),
        ("aime25", prompts(load("aime25"), "question")),
        ("amc23", prompts(load("amc23"), "question")),
        ("gpqa", prompts([gpqa_questions(gpqa)], "question")),
        ("gpqa_full", [x for k in ("Question", "Pre-Revision Question", "Extra Revised Question")
                       for x in prompts([gpqa], k)]),
        ("gsm8k", prompts(load("gsm8k"), "question")),
        ("humaneval", prompts([humaneval], "prompt")),
        ("ifbench", prompts(load("ifbench"), "prompt")),
        ("ifeval", prompts(load("ifeval"), "prompt")),
        ("lcb", prompts(load("lcb", lines=lambda r: {"q": r["question_content"]}), "q")),
        ("math500", prompts(load("math500"), "problem")),
        ("math_test", prompts(load("math_test"), "problem", per_file=True)),
        ("mbpp", prompts(mbpp, "text", per_file=True)),
        ("mmlu", prompts(mmlu, "question")),
        ("mmlu_q+choices", [(i, "prompt", r["question"] + "\n" + "\n".join(map(str, r["choices"])))
                            for i, r in enumerate(mmlu[0])]),
        ("ARC-Challenge", mc(load("arc_c")[0], lambda r: r["question"], choices)),
        ("ARC-Easy", mc(load("arc_e")[0], lambda r: r["question"], choices)),
        ("HellaSwag", mc(load("hellaswag")[0], lambda r: r["ctx"], lambda r: r["endings"])),
        ("MMLU-Redux", mc(mmlu_redux(), lambda r: r["question"], lambda r: r["choices"])),
        ("MMLU", mc(mmlu[0], lambda r: r["question"], lambda r: r["choices"])),
        ("OpenBookQA", mc(load("openbookqa")[0], lambda r: r["question_stem"], choices)),
        ("PIQA", mc(load("piqa")[0], lambda r: r["goal"], lambda r: [r["sol1"], r["sol2"]])),
        ("SIQA", mc(load("siqa")[0], lambda r: r["context"] + "\n" + r["question"],
                    lambda r: [r["answerA"], r["answerB"], r["answerC"]])),
        ("TruthfulQA", mc(load("truthfulqa")[0], lambda r: r["question"],
                          lambda r: r["mc1_targets"]["choices"])),
        ("WinoGrande", mc(load("winogrande")[0], lambda r: r["sentence"],
                          lambda r: [r["option1"], r["option2"]])),
        ("HumanEval", [(r["task_id"], p, r[f]) for r in humaneval
                       for p, f in (("canonical_solution", "canonical_solution"), ("prompt", "prompt"))]),
        ("MBPP", [(r["task_id"], "canonical_solution", r["code"]) for r in cat(mbpp)]),
    ]
    for bench, items in blocks:
        for i, part, text in items:
            yield bench, str(i), part, text
    for row in aime_rows:
        yield f"aime{int(row['year'])}", str(row['id']), "prompt", row['problem']


def write(out, aime90):
    from OnQ.evaluation.evalscope.prepare_aime90 import validate

    # Validate before downloading public sources. Reuse evaluation's schema and
    # counts so a partial 2026 file cannot silently pass as full coverage.
    with (Path(aime90) / 'default_test.jsonl').open(encoding='utf-8') as stream:
        aime_rows = [json.loads(line) for line in stream if line.strip()]
    validate(aime_rows)
    if any(not isinstance(row['problem'], str) for row in aime_rows):
        raise ValueError('AIME problem fields must be text')
    seen, rows = set(), []
    for key in build(aime_rows):
        if isinstance(key[3], str) and key[3].strip() and key not in seen:
            seen.add(key)
            rows.append(dict(zip(("benchmark", "id", "part", "text"), key)))
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(data)
    counts = collections.Counter((r["benchmark"], r["part"]) for r in rows)
    for (b, p), n in counts.items():
        print(f"{b:16s} {p:20s} {n:7d}")
    digest = hashlib.sha256(data).hexdigest()
    print(f"rows={len(rows)} labels={len({b for b, _ in counts})} sha256={digest}")


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def compare(built, frozen):
    same = Path(built).read_bytes() == Path(frozen).read_bytes()
    print(f"byte_identical={same}")
    ms = {}
    for tag, path in (("built", built), ("frozen", frozen)):
        ms[tag] = collections.defaultdict(collections.Counter)
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            ms[tag][(r["benchmark"], r["part"])][norm(r["text"])] += 1
    ok = True
    for key in sorted(set(ms["built"]) | set(ms["frozen"])):
        b, f = ms["built"][key], ms["frozen"][key]
        missing, extra = sum((f - b).values()), sum((b - f).values())
        ok &= not (missing or extra)
        print(f"{key[0]:16s} {key[1]:20s} frozen={sum(f.values()):6d} built={sum(b.values()):6d} "
              f"missing={missing} extra={extra}")
    print(f"normalized_multiset_identical={ok}")
    return same or ok


def main():
    global CACHE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--aime90", type=Path, required=True,
                    help="Prepared AIME 2024-2026 evaluation directory containing default_test.jsonl")
    ap.add_argument("--cache", help="download cache directory (HF files and archives)")
    ap.add_argument("--compare", help="frozen HELDOUT_QUERIES.jsonl to compare against")
    a = ap.parse_args()
    CACHE = a.cache
    write(a.out, a.aime90)
    if a.compare and not compare(a.out, a.compare):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
