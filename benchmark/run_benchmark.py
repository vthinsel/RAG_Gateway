import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parent
DEFAULT_OPTIONS = {
    "top_k": 5,
    "hybrid_enabled": False,
    "rerank_enabled": False,
    "retrieval_min_score": 0.72,
    "retrieval_num_candidates": 100,
    "temperature": 0.0,
    "reasoning_effort": "none",
}
SERVER_FIELDS = {
    "ollama_model", "ollama_reasoning_effort", "public_model_name", "temperature",
    "voyage_embedding_model", "embedding_dimensions", "mongodb_database",
    "mongodb_collection", "atlas_vector_index", "atlas_search_index",
    "retrieval_top_k", "retrieval_num_candidates", "retrieval_min_score",
    "retrieval_rerank_enabled", "retrieval_rerank_model", "retrieval_rerank_candidates",
    "retrieval_hybrid_enabled", "retrieval_hybrid_candidates", "max_query_chars",
    "upstream_timeout_seconds", "chunk_size", "chunk_overlap",
}
INTEGER_OPTIONS = {"top_k": (1, 50), "retrieval_num_candidates": (1, None),
                   "max_tokens": (1, None), "seed": (None, None)}
FLOAT_OPTIONS = {"temperature": (0, 2), "retrieval_min_score": (0, 1),
                 "top_p": (0, 1), "presence_penalty": (-2, 2),
                 "frequency_penalty": (-2, 2)}
OPTION_NAMES = set(INTEGER_OPTIONS) | set(FLOAT_OPTIONS) | {
    "hybrid_enabled", "rerank_enabled", "reasoning_effort", "stop", "response_format",
}
RETRIEVAL_OPTIONS = {
    "top_k", "hybrid_enabled", "rerank_enabled", "retrieval_min_score",
    "retrieval_num_candidates",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def validate_options(options: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(options, dict):
        raise TypeError("Options must be a JSON object")
    unknown = set(options) - OPTION_NAMES
    if unknown:
        raise ValueError(f"Not supported at run time: {', '.join(sorted(unknown))}")
    result = {**DEFAULT_OPTIONS, **options}
    for name, value in result.items():
        if name in INTEGER_OPTIONS or name in FLOAT_OPTIONS:
            bounds = INTEGER_OPTIONS.get(name, FLOAT_OPTIONS.get(name))
            valid_type = type(value) is int if name in INTEGER_OPTIONS else type(value) in {int, float}
            if not valid_type or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite numeric value of the correct type")
            minimum, maximum = bounds
            if (minimum is not None and value < minimum) or (
                maximum is not None and value > maximum
            ):
                raise ValueError(f"{name} is outside its supported range")
        elif name in {"hybrid_enabled", "rerank_enabled"} and type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
        elif name == "reasoning_effort":
            if not isinstance(value, str) or len(value) > 64 or any(
                character in value for character in "\r\n"
            ):
                raise ValueError("reasoning_effort must be a short single-line string")
            result[name] = value.strip()
        elif name == "stop" and not (
            isinstance(value, str) or isinstance(value, list)
            and all(isinstance(item, str) for item in value)
        ):
            raise ValueError("stop must be a string or array of strings")
        elif name == "response_format" and not isinstance(value, dict):
            raise ValueError("response_format must be an object")
    return result


def load_questions(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    document = read_json(path)
    questions = document.get("questions") if isinstance(document, dict) else None
    if not isinstance(questions, list) or not questions:
        raise ValueError("Questions file must contain a nonempty questions array")
    rows = []
    seen = set()
    for row in questions:
        if not isinstance(row, dict) or set(row) - {"id", "category", "question"}:
            raise ValueError("Use the blind questions-only file, not the answer key")
        if any(not isinstance(row.get(name), str) or not row[name].strip()
               for name in ("id", "question")):
            raise ValueError("Each question needs a nonblank string id and question")
        if row["id"] in seen:
            raise ValueError(f"Duplicate question id: {row['id']}")
        seen.add(row["id"])
        rows.append({"id": row["id"], "category": row.get("category", ""),
                     "question": row["question"]})
    if document.get("question_count", len(rows)) != len(rows):
        raise ValueError("question_count does not match the questions array")
    metadata = {name: document.get(name) for name in ("name", "corpus_archive", "snapshot_date")}
    metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["question_count"] = len(rows)
    return metadata, rows


def make_payload(question: str, options: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "retrieve":
        return {"query": question, **{key: value for key, value in options.items()
                                     if key in RETRIEVAL_OPTIONS}}
    return {"messages": [{"role": "user", "content": question}],
            "stream": False, **options}


def normalize_response(response: dict[str, Any], mode: str) -> dict[str, Any]:
    choices = response.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    answer = message.get("content") or ""
    sources = response.get("data", []) if mode == "retrieve" else response.get("sources", [])
    citations = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", answer))) if isinstance(answer, str) else []
    citations_available = {source.get("citation") for source in sources}
    usage = response.get("usage") or {}
    prompt = usage.get("prompt_tokens", response.get("prompt_eval_count"))
    completion = usage.get("completion_tokens", response.get("eval_count"))
    total = usage.get("total_tokens")
    if total is None and type(prompt) is int and type(completion) is int:
        total = prompt + completion
    return {
        "answer": answer if mode == "chat" else None,
        "reasoning": message.get("reasoning_content", message.get("reasoning")),
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "reported_model": response.get("model"),
        "sources": sources,
        "citations_in_answer": citations,
        "invalid_citation_labels": [label for label in citations if label not in citations_available],
        "tokens": {"prompt": prompt, "completion": completion, "total": total},
        "raw_usage": usage,
        "upstream_timings": {key: response[key] for key in (
            "total_duration", "load_duration", "prompt_eval_duration", "eval_duration"
        ) if key in response},
    }


async def perform_request(
    client: httpx.AsyncClient, endpoint: str, payload: dict[str, Any], mode: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {"endpoint": endpoint, "request_payload": payload,
                              "started_at": utc_now(), "status": "error"}
    try:
        response = await client.post(endpoint, json=payload)
        result["http_status"] = response.status_code
        result["response_headers"] = {name: response.headers[name] for name in (
            "x-request-id", "x-correlation-id", "server-timing", "retry-after"
        ) if name in response.headers}
        try:
            raw = response.json()
        except ValueError:
            raw = None
        result["raw_response"] = raw
        if response.is_success and isinstance(raw, dict):
            if mode == "chat" and not raw.get("choices"):
                raise ValueError("Successful chat response has no choices")
            if mode == "retrieve" and not isinstance(raw.get("data"), list):
                raise ValueError("Successful retrieval response has no data array")
            result.update(normalize_response(raw, mode))
            result["status"] = "ok"
        else:
            result["error"] = {"type": "HTTPError", "message": f"HTTP {response.status_code}",
                               "body": raw if raw is not None else response.text}
    except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError) as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        result["finished_at"] = utc_now()
        result["timings_ms"] = {"client_total": round((time.perf_counter() - started) * 1000, 3),
                                "ttft": None, "embedding": None, "search": None,
                                "reranking": None, "generation": None}
    return result


def output_path(directory: str, filename: str) -> Path:
    folder = (ROOT / directory).resolve()
    if not folder.is_relative_to(ROOT):
        raise ValueError("Generated files must stay inside the benchmark folder")
    return folder / filename


def redact(value: Any, secret: str) -> Any:
    if isinstance(value, dict):
        return {key: redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "[REDACTED]")
    return value


def save_result(path: Path, result: dict[str, Any], secret: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(redact(result, secret), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in records if not row["warmup"]]
    successful = [row for row in measured if row["status"] == "ok"]
    durations = [row["timings_ms"]["client_total"] for row in successful]
    tokens = [row.get("tokens", {}).get("total") for row in successful]
    reported = [value for value in tokens if type(value) is int]
    return {
        "completed": len(measured), "ok": len(successful),
        "errors": len(measured) - len(successful),
        "warmup_completed": len(records) - len(measured),
        "mean_client_total_ms": statistics.mean(durations) if durations else None,
        "median_client_total_ms": statistics.median(durations) if durations else None,
        "reported_total_tokens": sum(reported) if reported else None,
        "responses_with_token_counts": len(reported),
        "responses_without_token_counts": len(successful) - len(reported),
        "empty_source_responses": sum(not row.get("sources") for row in successful),
        "invalid_citation_count": sum(len(row.get("invalid_citation_labels", []))
                                      for row in successful),
        "retrieval_probe_errors": sum(row.get("retrieval_probe", {}).get("status") == "error"
                                      for row in records),
    }


def describe_conditions(options: dict[str, Any], settings: dict[str, Any], mode: str) -> dict:
    hybrid = options["hybrid_enabled"]
    rerank = options["rerank_enabled"]
    top_k = options["top_k"]
    rerank_limit = settings.get("retrieval_rerank_candidates")
    hybrid_limit = settings.get("retrieval_hybrid_candidates")
    vector_limit = top_k
    if hybrid or rerank:
        limits = [max(value, top_k) if type(value) is int else None
                  for value in ([hybrid_limit, rerank_limit] if hybrid and rerank
                                else [hybrid_limit if hybrid else rerank_limit])]
        vector_limit = min(limits) if all(value is not None for value in limits) else None
    warnings = []
    if hybrid:
        warnings.append("retrieval_min_score is ignored by hybrid search in this gateway version")
    if options.get("reasoning_effort") == "":
        warnings.append("Empty reasoning_effort omits the upstream field; model default may reason")
    if mode == "retrieve":
        warnings.append("Generation options are not sent in retrieval-only mode")
    warnings.extend([
        "Reasoning and generation-option support is model-specific and not confirmed by HTTP success",
        "Reported response model is a gateway alias, not proof of the actual Ollama model",
        "Server settings, when supplied, describe configuration; pending restart may cause runtime drift",
    ])
    return {
        "requested_options": options,
        "forced_request_fields": sorted(make_payload("", options, mode).keys()),
        "generation_options_not_sent": sorted(set(options) - RETRIEVAL_OPTIONS)
        if mode == "retrieve" else [],
        "unforced_generation_fields": sorted({"top_p", "max_tokens", "seed", "stop",
                                               "presence_penalty", "frequency_penalty",
                                               "response_format"} - set(options)),
        "configured_actual_model": settings.get("ollama_model"),
        "score_kind": "reciprocal_rank_fusion" if hybrid else "atlas_vector_score",
        "score_cutoff_applied": not hybrid,
        "vector_limit_from_config": vector_limit,
        "num_candidates_from_config": max(options["retrieval_num_candidates"], vector_limit)
        if vector_limit is not None else None,
        "server_only_settings": {name: settings.get(name) for name in sorted(SERVER_FIELDS)},
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blind RAG Gateway benchmark client")
    parser.add_argument("--base-url", help="Gateway root URL, without /v1")
    parser.add_argument("--tenant", required=True, help="Actual ingested tenant ID")
    parser.add_argument("--questions", type=Path,
                        default=ROOT / "homelab_benchmark_v16_questions_only.json")
    parser.add_argument("--matrix", type=Path, help="Named profiles with options")
    parser.add_argument("--options-file", type=Path, help="Shared runtime-options JSON object")
    parser.add_argument("--name", default="single", help="Single-profile name")
    parser.add_argument("--mode", choices=["chat", "retrieve"], default="chat")
    parser.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--reasoning", help="Explicit level, e.g. none or high; model-specific")
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--seed", type=int, help="Upstream generation seed")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0, help="Excluded requests per profile")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--order-seed", type=int, default=42)
    parser.add_argument("--ids", help="Comma-separated question IDs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--capture-retrieval", action="store_true",
                        help="Separate post-chat retrieval request for chunk text")
    parser.add_argument("--capture-settings", action="store_true",
                        help="Read configured nonsecret settings; requires administrator key")
    parser.add_argument("--server-settings", type=Path,
                        help="Operator-provided settings JSON, filtered to nonsecret fields")
    parser.add_argument("--document-map", type=Path,
                        help="Verified filename/frontmatter-ID to actual Atlas IDs mapping")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--progress-interval", type=float, default=5)
    parser.add_argument("--token-env", default="GATEWAY_API_KEY")
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument("--output-dir", default="results", help="Directory within benchmark/")
    parser.add_argument("--dry-run", action="store_true", help="Write plans without any HTTP requests")
    return parser


def load_profiles(args: argparse.Namespace) -> list[dict[str, Any]]:
    shared = read_json(args.options_file) if args.options_file else {}
    validate_options(shared)
    aliases = {"hybrid": "hybrid_enabled", "rerank": "rerank_enabled",
               "min_score": "retrieval_min_score", "num_candidates": "retrieval_num_candidates",
               "reasoning": "reasoning_effort"}
    for flag in ("hybrid", "rerank", "top_k", "min_score", "num_candidates",
                 "temperature", "reasoning", "top_p", "max_tokens", "seed"):
        value = getattr(args, flag)
        if value is not None:
            shared[aliases.get(flag, flag)] = value
    document = read_json(args.matrix) if args.matrix else {"profiles": [{"name": args.name}]}
    profiles = document.get("profiles") if isinstance(document, dict) else None
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Matrix must have a nonempty profiles array")
    names = set()
    result = []
    for profile in profiles:
        if not isinstance(profile, dict) or set(profile) - {"name", "options", "description"}:
            raise ValueError("Each profile supports only name, description and options")
        name = profile.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("Profile names must be unique safe ASCII names")
        options = profile.get("options", {})
        validate_options(options)
        names.add(name)
        result.append({"name": name, "description": profile.get("description", ""),
                       "options": validate_options({**shared, **options})})
    return result


async def with_progress(operation: Any, label: str, interval: float) -> Any:
    task = asyncio.create_task(operation)
    started = time.perf_counter()
    try:
        while True:
            completed, _pending = await asyncio.wait({task}, timeout=interval)
            if completed:
                return task.result()
            print(f"  waiting {label} | {time.perf_counter() - started:.1f}s elapsed", flush=True)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def execute_profiles(
    args: argparse.Namespace, profiles: list[dict], dataset: dict, questions: list[dict],
    client: httpx.AsyncClient | None, secret: str,
) -> list[dict]:
    settings = {}
    settings_source = "unknown"
    if args.server_settings:
        supplied = read_json(args.server_settings)
        supplied = supplied.get("settings", supplied)
        settings = {key: value for key, value in supplied.items() if key in SERVER_FIELDS}
        settings_source = "operator_file_not_runtime_verified"
    if args.capture_settings and not args.dry_run:
        response = await client.get("/v1/admin/environment")
        response.raise_for_status()
        settings = {key: value for key, value in response.json()["settings"].items()
                    if key in SERVER_FIELDS}
        settings_source = "admin_configured_settings_not_runtime_verified"
    mapping = read_json(args.document_map) if args.document_map else {}
    if not isinstance(mapping, dict) or any(
        not isinstance(key, str) or not isinstance(values, list)
        or not all(isinstance(value, str) and value.strip() for value in values)
        for key, values in mapping.items()
    ):
        raise ValueError("Document map must map names to arrays of verified Atlas IDs")
    run_id = uuid.uuid4().hex[:12]
    results = []
    endpoint = "/v1/chat/completions" if args.mode == "chat" else "/v1/retrieve"
    for profile in profiles:
        rng = random.Random(args.order_seed)
        plan = []
        for index in range(args.warmup):
            plan.append({"question": questions[index % len(questions)], "repetition": 0,
                         "warmup": True})
        for repetition in range(1, args.repetitions + 1):
            ordered = list(questions)
            if args.shuffle:
                rng.shuffle(ordered)
            plan.extend({"question": question, "repetition": repetition, "warmup": False}
                        for question in ordered)
        path = output_path(args.output_dir, f"{run_id}_{profile['name']}.json")
        result = {
            "schema_version": "rag-gateway-benchmark/1", "run_id": run_id,
            "profile": profile["name"], "description": profile.get("description", ""),
            "state": "planned" if args.dry_run else "running", "started_at": utc_now(),
            "finished_at": None, "dataset": dataset,
            "conditions": {
                **describe_conditions(profile["options"], settings, args.mode),
                "base_url": args.base_url, "tenant": args.tenant, "mode": args.mode,
                "settings_source": settings_source, "repetitions": args.repetitions,
                "warmup_requests": args.warmup, "shuffle": args.shuffle,
                "order_seed": args.order_seed, "timeout_seconds": args.timeout,
                "concurrency": 1, "automatic_retries": 0, "stream": False,
                "capture_retrieval": args.capture_retrieval,
                "tls_verification": True, "custom_ca": bool(args.ca_bundle),
                "client_python": platform.python_version(), "client_os": platform.platform(),
                "httpx_version": httpx.__version__,
                "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            "document_id_mapping": mapping,
            "availability": {
                "chat_context_text": "not_exposed_by_chat_endpoint",
                "retrieval_probe": "separate_request_not_exact_chat_context"
                if args.capture_retrieval else "not_requested",
                "ttft_and_stage_timings": "not_exposed_by_nonstreaming_gateway",
                "pre_rerank_and_prefusion_ranks": "not_exposed_by_gateway",
                "missing_metrics": "null_means_unavailable_not_zero",
            },
            "execution_plan": [{"question_id": item["question"]["id"],
                                "repetition": item["repetition"], "warmup": item["warmup"]}
                               for item in plan],
            "records": [], "summary": summary([]),
        }
        save_result(path, result, secret)
        print(f"Profile {profile['name']} | {len(plan)} requests | {path}", flush=True)
        if args.dry_run:
            results.append(result)
            continue
        try:
            for sequence, item in enumerate(plan, start=1):
                question = item["question"]
                label = f"{profile['name']} {question['id']} rep={item['repetition']}"
                print(f"[{sequence}/{len(plan)}] {label} | {'warmup' if item['warmup'] else 'request'}",
                      flush=True)
                record = {
                    "question_id": question["id"], "category": question["category"],
                    "question": question["question"], "repetition": item["repetition"],
                    "warmup": item["warmup"], "sequence": sequence,
                    **await with_progress(perform_request(
                        client, endpoint, make_payload(question["question"], profile["options"],
                                                       args.mode), args.mode,
                    ), label, args.progress_interval),
                }
                result["records"].append(record)
                result["summary"] = summary(result["records"])
                save_result(path, result, secret)
                if args.capture_retrieval and args.mode == "chat" and record["status"] == "ok":
                    print(f"  {label} | separate retrieval probe", flush=True)
                    probe = await with_progress(perform_request(
                        client, "/v1/retrieve",
                        make_payload(question["question"], profile["options"], "retrieve"), "retrieve",
                    ), f"{label} retrieval-probe", args.progress_interval)
                    probe["is_exact_chat_context"] = False
                    chat_keys = [(source.get("document_id"), source.get("chunk_index"))
                                 for source in record.get("sources", [])]
                    probe_keys = [(source.get("document_id"), source.get("chunk_index"))
                                  for source in probe.get("sources", [])]
                    probe["same_document_chunk_order_as_chat"] = chat_keys == probe_keys
                    record["retrieval_probe"] = probe
                    result["summary"] = summary(result["records"])
                    save_result(path, result, secret)
                metrics = result["summary"]
                print(f"  {record['status']} | {record['timings_ms']['client_total']:.0f}ms"
                      f" | sources={len(record.get('sources', []))}"
                      f" | tokens={record.get('tokens', {}).get('total')}"
                      f" | completed={metrics['completed']} errors={metrics['errors']}"
                      f" | mean={metrics['mean_client_total_ms']}ms", flush=True)
            result["state"] = "completed_with_errors" if result["summary"]["errors"] else "completed"
        except (KeyboardInterrupt, asyncio.CancelledError):
            result["state"] = "interrupted"
            raise
        finally:
            result["finished_at"] = utc_now()
            save_result(path, result, secret)
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.tenant.strip() or len(args.tenant) > 256:
            raise ValueError("Tenant must be nonblank and at most 256 characters")
        if args.repetitions < 1 or args.warmup < 0 or (args.limit is not None and args.limit < 1):
            raise ValueError("Repetitions/limit must be positive; warmup must be nonnegative")
        if not all(math.isfinite(value) and value > 0 for value in (
            args.timeout, args.progress_interval
        )):
            raise ValueError("Timeout and progress interval must be finite and positive")
        if not args.dry_run and not args.base_url:
            raise ValueError("--base-url is required for live runs")
        if args.base_url:
            url = urlsplit(args.base_url)
            if url.scheme not in {"http", "https"} or not url.hostname or (
                url.username or url.password or url.query or url.fragment
            ) or url.path.rstrip("/").endswith("/v1"):
                raise ValueError("Use a gateway root http(s) URL without credentials, query, or /v1")
        output_path(args.output_dir, "check.json")
        dataset, questions = load_questions(args.questions)
        if args.ids:
            selected = set(args.ids.split(","))
            unknown = selected - {row["id"] for row in questions}
            if unknown:
                raise ValueError(f"Unknown question IDs: {sorted(unknown)}")
            questions = [row for row in questions if row["id"] in selected]
        if args.limit:
            questions = questions[:args.limit]
        profiles = load_profiles(args)
        secret = ""
        if not args.dry_run:
            secret = os.getenv(args.token_env, "").strip()
            if not secret:
                if not sys.stdin.isatty():
                    raise ValueError(f"Set {args.token_env} in the environment before running")
                import getpass
                secret = getpass.getpass("Gateway bearer token (hidden): ").strip()
            if not secret or any(character in secret for character in "\r\n"):
                raise ValueError("Gateway token must be a nonblank single-line value")
        async def execute() -> list[dict]:
            if args.dry_run:
                return await execute_profiles(args, profiles, dataset, questions, None, "")
            import ssl
            verification = ssl.create_default_context(cafile=str(args.ca_bundle)) \
                if args.ca_bundle else True
            async with httpx.AsyncClient(
                base_url=args.base_url.rstrip("/") + "/", timeout=args.timeout,
                headers={"Authorization": f"Bearer {secret}", "X-Tenant-ID": args.tenant},
                verify=verification, follow_redirects=False,
            ) as client:
                return await execute_profiles(args, profiles, dataset, questions, client, secret)
        results = asyncio.run(execute())
        return 1 if any(result["state"] == "completed_with_errors" for result in results) else 0
    except KeyboardInterrupt:
        print("Interrupted. Completed records were checkpointed.", file=sys.stderr)
        return 130
    except (ValueError, TypeError, OSError, httpx.HTTPError) as error:
        print(f"Benchmark error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())