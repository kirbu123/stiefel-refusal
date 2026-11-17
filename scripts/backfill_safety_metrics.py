#!/usr/bin/env python3
"""Backfill answer quality from RDO response logs, without running generation."""
from __future__ import annotations

import argparse
import copy
import csv
import io
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
# evaluate.__init__ imports unrelated API judges. Keep dry-run and CPU checks
# independent of their dependencies, just as the quality evaluator itself is.
_spec = importlib.util.spec_from_file_location(
    "backfill_safety_evaluator", ROOT / "evaluate/evaluation_safety_analisys.py")
safety = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(safety)

STAMP = re.compile(r"(\d{8}_\d{6})\.json$")
PHASE_FILES = {"initial": "initial_responses", "refined_attack": "refined_responses_attack",
               "refined_protect": "refined_responses_protect"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path, text):
    """Replace one artifact atomically; subsequent runs repair derived artifacts."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, data):
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def required_phases(payload):
    phases = {"initial"}
    blocks = list((payload.get("guard_metrics") or {}).values())
    blocks += [payload.get("llamaguard") or {}]
    blocks += [source.get("results", {}) for source in
               (payload.get("safety_analisys") or {}).get("splits", {}).values()]
    for block in blocks:
        for kind in safety.KINDS:
            for phase, stats in (block.get(kind) or {}).items():
                if stats:
                    phases.add("refined_attack" if phase == "refined" else phase)
    return phases & PHASE_FILES.keys()


def load_experiment(directory):
    candidates = sorted(p for p in directory.glob("eval_metrics_*.json") if STAMP.search(p.name))
    if not candidates:
        raise ValueError(f"No eval_metrics_*.json in {directory}")
    metrics_path = candidates[-1]
    if not metrics_path.with_suffix(".csv").exists():
        raise ValueError(f"Missing metrics CSV: {metrics_path.with_suffix('.csv')}")
    cutoff = STAMP.search(metrics_path.name).group(1)
    payload = read_json(metrics_path)
    hp_path = directory / "hparams.json"
    hp = read_json(hp_path) if hp_path.exists() else {}
    generations, files = {}, {}
    for split, mode in (("train", "basic"), ("test", "rdo")):
        folder = directory / "generated_responses" / f"{split}_answers"
        logs = [p for p in folder.glob("*.json") if STAMP.search(p.name)
                and any(p.name == f"{prefix}_{STAMP.search(p.name).group(1)}.json"
                        for prefix in PHASE_FILES.values())
                and STAMP.search(p.name).group(1) <= cutoff]
        if not logs:
            raise ValueError(f"Missing response logs: {folder}")
        stamp = max(STAMP.search(p.name).group(1) for p in logs)
        phases = required_phases(payload) | {
            phase for phase, prefix in PHASE_FILES.items()
            if (folder / f"{prefix}_{stamp}.json").exists()}
        questions, responses = {}, {}
        files[split] = {}
        for phase in PHASE_FILES:
            if phase not in phases:
                continue
            path = folder / f"{PHASE_FILES[phase]}_{stamp}.json"
            data = read_json(path)
            config = data.get("config", {})
            if config.get("llamaguard_data_mode", mode) != mode:
                raise ValueError(f"Wrong dataset mode: {path}")
            responses[phase] = {}
            for kind in safety.KINDS:
                pairs = data.get(kind)
                if not isinstance(pairs, list) or not pairs or any(
                    not isinstance(row, dict)
                    or not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
                    or not isinstance(row.get("response"), str) for row in pairs
                ):
                    raise ValueError(f"Invalid {kind} response records: {path}")
                prompts = [row["prompt"] for row in pairs]
                if kind in questions and questions[kind] != prompts:
                    raise ValueError(f"Prompt mismatch: {split}/{kind}/{phase}")
                questions[kind] = prompts
                responses[phase][kind] = [row["response"] for row in pairs]
            files[split][phase] = str(path.relative_to(directory))
        generations[split] = {
            "source": {"mode": mode,
                       "eval_split": payload.get("eval_split", hp.get("eval_split")) if mode == "rdo" else None,
                       "splits_name": hp.get("splits") if mode == "rdo" else None,
                       "model_name": hp.get("model") if mode == "basic" else None},
            "questions": questions, "phases": responses}
    return metrics_path, payload, generations, files


def complete(stats, n):
    if not isinstance(stats, dict) or stats.get("n") != n:
        return False
    mean, counts = stats.get("mean_score"), stats.get("score_counts", {})
    return (isinstance(mean, (int, float)) and math.isfinite(mean) and 1 <= mean <= 4
            and isinstance(counts, dict) and set(counts) == {"1", "2", "3", "4"}
            and all(type(v) is int and v >= 0 for v in counts.values())
            and sum(counts.values()) == n
            and math.isclose(mean, sum(int(k) * v for k, v in counts.items()) / n))


def missing_groups(payload, generations):
    splits = (payload.get("safety_analisys") or {}).get("splits", {})
    return [(split, kind, phase) for split, data in generations.items()
            for kind in safety.KINDS for phase in data["phases"]
            if not complete(splits.get(split, {}).get("results", {}).get(kind, {}).get(phase),
                            len(data["questions"][kind]))]


def update_csv(path, block):
    if not path.exists():
        raise ValueError(f"Missing metrics CSV (cannot preserve other metrics): {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = [row for row in reader if row.get("backend") != "safetyanalisys"]
    for field in ("benchmark", "backend", "group", "split", "phase", "metric", "value"):
        if field not in fields:
            fields.append(field)
    rows.extend(safety.metrics_to_csv_rows(block))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def publish_tensorboard(directory, block):
    # Read actual events so retries after interrupted publication don't duplicate scalars.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    import time

    accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0},
                                   purge_orphaned_data=False)
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    pending = []
    for row in safety.metrics_to_csv_rows(block):
        if row["metric"] != "mean_score":
            continue
        tag = f"eval/guard/safetyanalisys/{row['split']}/{row['group']}/{row['phase']}/mean_score"
        events = accumulator.Scalars(tag) if tag in tags else []
        if not any(e.step == 0 and math.isclose(e.value, row["value"], rel_tol=1e-6) for e in events):
            pending.append((tag, row["value"]))
    if pending:
        writer = EventFileWriter(str(directory))
        try:
            for tag, value in pending:
                writer.add_event(Event(wall_time=time.time(), step=0,
                                       summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)])))
            writer.flush()
        finally:
            writer.close()


def process(directory, args, get_evaluator):
    path, payload, generations, files = load_experiment(directory)
    missing = missing_groups(payload, generations)
    total = sum(len(d["phases"]) * len(safety.KINDS) for d in generations.values())
    print(f"{directory}: {path.name}; missing={len(missing)}, ready={total - len(missing)}")
    for split, phases in files.items():
        print(f"  {split}: {', '.join(phases.values())}")
    if args.dry_run:
        for group in missing:
            print("  missing: " + "/".join(group))
        return len(missing), total - len(missing)

    payload = copy.deepcopy(payload)
    block = payload["safety_analisys"] = payload.get("safety_analisys") or {}
    details_path = directory / block.get("scores_file", f"safety_analisys_scores_backfill_{STAMP.search(path.name).group(1)}.json")
    details = read_json(details_path).get("scores", []) if details_path.exists() else []
    if missing:
        if not args.safety_analisys_references:
            raise ValueError("Missing groups require --safety-analisys-references")
        _, digest = safety.load_references(args.safety_analisys_references)
        expected = {"model": safety.MODEL_NAME, "prompt_version": safety.PROMPT_VERSION,
                    "references_sha256": digest}
        if total != len(missing) and any(block.get("config", {}).get(k) != v for k, v in expected.items()):
            raise ValueError("Existing SafetyAnalisys judge/references configuration is incompatible")
        evaluator = get_evaluator()
        for split, kind, phase in missing:
            data = generations[split]
            pairs = list(zip(data["questions"][kind], data["phases"][phase][kind]))
            print(f"  scoring {split}/{kind}/{phase}: n={len(pairs)}", flush=True)
            results = evaluator.evaluate_batch(pairs, kind=kind)
            if len(results) != len(pairs):
                raise ValueError("Judge returned incomplete results")
            stats = safety.aggregate(results)
            if not complete(stats, len(pairs)):
                raise ValueError("Judge returned invalid scores")
            split_block = block.setdefault("splits", {}).setdefault(split, {"results": {}})
            split_block["source"] = data["source"]
            split_block.setdefault("results", {}).setdefault(kind, {})[phase] = stats
            details = [r for r in details if (r["split"], r["kind"], r["phase"]) != (split, kind, phase)]
            details.extend(dict(split=split, kind=kind, phase=phase, prompt=q, response=a, **r)
                           for (q, a), r in zip(pairs, results))
        block["config"] = evaluator.config
        block["scores_file"] = details_path.name
    block["response_files"] = files
    payload.get("guard_errors", {}).pop("safetyanalisys", None)
    csv_path = path.with_suffix(".csv")
    csv_text = update_csv(csv_path, block)
    # Judge failures above leave all existing artifacts intact. JSON is the recovery source.
    if missing:
        write_json(details_path, {"config": block["config"], "scores": details})
    if payload != read_json(path):
        write_json(path, payload)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        previous_csv = stream.read()
    if csv_text != previous_csv:
        atomic_write(csv_path, csv_text)
    publish_tensorboard(directory, block)
    return len(missing), total - len(missing)


def maybe_launch_ppu(args):
    """Use the same scheduler/container boundary as run_rdo_refusal.sh."""
    platform = os.environ.get("RDO_PLATFORM", "auto")
    if (args.dry_run or os.environ.get("RDO_PPU_CONTAINER", "0") == "1"
            or os.environ.get("RDO_DEVICE", "auto") == "cpu"):
        return
    if platform != "ppu" and not (platform == "auto" and Path("/dev/alixpu").exists()):
        return
    # The launcher changes to the repository root. Resolve user paths before
    # crossing that boundary so paths containing spaces and other cwd's work.
    forwarded = ["--exp-dirs", *(str(p.expanduser().resolve()) for p in args.exp_dirs),
                 "--metrics", *args.metrics,
                 "--safety-analisys-references", str(args.safety_analisys_references.expanduser().resolve()),
                 "--safety-analisys-batch-size", str(args.safety_analisys_batch_size)]
    env = dict(os.environ, RDO_REPO_ROOT=str(ROOT))
    os.execvpe("bash", ["bash", str(ROOT / "scripts/launch_ppu.sh"), "backfill", *forwarded], env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", choices=["safetyanalisys"], default=["safetyanalisys"])
    parser.add_argument("--safety-analisys-references", type=Path,
                        default=safety.DEFAULT_REFERENCES_PATH,
                        help=f"References JSON (default: {safety.DEFAULT_REFERENCES_PATH})")
    parser.add_argument("--safety-analisys-batch-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.safety_analisys_batch_size < 1:
        parser.error("--safety-analisys-batch-size must be positive")
    maybe_launch_ppu(args)
    evaluator = None

    def get_evaluator():
        nonlocal evaluator
        if evaluator is None:
            evaluator = safety.SafetyAnalisysEvaluator(args.safety_analisys_references,
                                                       args.safety_analisys_batch_size)
        return evaluator

    failed = calculated = skipped = 0
    try:
        for directory in dict.fromkeys(p.expanduser().resolve() for p in args.exp_dirs):
            try:
                done, ready = process(directory, args, get_evaluator)
                calculated += done
                skipped += ready
                print(f"  {'DRY-RUN' if args.dry_run else 'OK'}")
            except Exception as error:
                failed += 1
                print(f"ERROR {directory}: {type(error).__name__}: {error}", file=sys.stderr)
    finally:
        if evaluator is not None:
            evaluator.close()
    print(f"{'Planned' if args.dry_run else 'Calculated'} groups: {calculated}; ready: {skipped}; failed experiments: {failed}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
