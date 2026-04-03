import json
import os
from functools import lru_cache
from typing import Any

from plugins.location_tools.repo_ops.repo_ops import (
    get_current_issue_id,
    get_current_repo_modules,
)


VT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "vt_results")
MAX_ARTIFACT_LIMIT = 12
MAX_DEP_LIMIT = 16
MAX_PATH_LIMIT = 6
MAX_SYMBOL_SUMMARY = 12


def _require_current_instance_id() -> str:
    instance_id = get_current_issue_id()
    if not instance_id:
        raise RuntimeError(
            "No current issue is set. DaLIC raw tools can only be used while localizing an active instance."
        )
    return instance_id


def _compact_external_path(path: str) -> str:
    norm_path = path.replace("\\", "/").strip()
    parts = [part for part in norm_path.split("/") if part]
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return norm_path


def _normalize_repo_path(path: str) -> str:
    norm_path = path.replace("\\", "/").strip()
    if not norm_path:
        return norm_path

    repo_modules = get_current_repo_modules()
    if repo_modules and repo_modules[0]:
        repo_files = [file_info["name"] for file_info in repo_modules[0]]
        suffix_matches = [
            candidate
            for candidate in repo_files
            if norm_path == candidate or norm_path.endswith("/" + candidate)
        ]
        if suffix_matches:
            return max(suffix_matches, key=len)

    return _compact_external_path(norm_path)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _contains_casefold(haystack: str, needle: str | None) -> bool:
    if needle is None:
        return True
    return needle.casefold() in haystack.casefold()


def _clamp_int(value: int, default: int, maximum: int) -> int:
    if value is None:
        value = default
    value = int(value)
    if value < 0:
        value = 0
    return min(value, maximum)


@lru_cache(maxsize=128)
def _load_unique_artifacts(instance_id: str) -> dict:
    file_path = os.path.join(VT_RESULTS_DIR, instance_id, "unique_artifacts.json")
    with open(file_path, "r") as file:
        return json.load(file)


@lru_cache(maxsize=128)
def _load_unique_dep_tree(instance_id: str) -> dict:
    file_path = os.path.join(VT_RESULTS_DIR, instance_id, "unique_dep_tree.json")
    with open(file_path, "r") as file:
        return json.load(file)


def _artifact_matches_type(artifact: dict, artifact_type: str) -> bool:
    definition_type = _normalize_text(artifact.get("definition_type")).lower()
    class_name = _normalize_text(artifact.get("class_name"))

    if artifact_type == "all":
        return True
    if artifact_type == "class":
        return definition_type == "class"
    if artifact_type == "method":
        return definition_type == "function" and bool(class_name)
    if artifact_type == "function":
        return definition_type == "function" and not class_name
    return True


def get_trace_artifacts(
    artifact_type: str = "all",
    path_contains: str | None = None,
    name_contains: str | None = None,
    start: int = 0,
    limit: int = 8,
):
    """Return a compact raw slice of DaLIC execution-trace artifacts for the current instance.

    This tool reads `unique_artifacts.json` from `DaLIC/vt_results/<instance_id>/`.
    It does not rank or recommend results. It only applies exact filters and pagination,
    then returns a compact JSON summary.

    Tips:
    - Use this once near the start of analyzing a new instance to inspect the raw execution-trace evidence.
    - You can call it again later with narrower `path_contains` or `name_contains` filters whenever you need a more precise slice.
    """
    instance_id = _require_current_instance_id()
    artifact_type = _normalize_text(artifact_type).lower() or "all"
    start = _clamp_int(start, default=0, maximum=10_000)
    limit = _clamp_int(limit, default=8, maximum=MAX_ARTIFACT_LIMIT)

    artifact_data = _load_unique_artifacts(instance_id)
    raw_artifacts = artifact_data.get("unique_functions", [])
    filtered_artifacts = []
    for artifact in raw_artifacts:
        normalized_path = _normalize_repo_path(_normalize_text(artifact.get("path")))
        qualified_name = _normalize_text(
            artifact.get("qualified_name") or artifact.get("name")
        )
        if not _artifact_matches_type(artifact, artifact_type):
            continue
        if not _contains_casefold(normalized_path, path_contains):
            continue
        if not _contains_casefold(qualified_name, name_contains):
            continue
        filtered_artifacts.append(
            {
                "qualified_name": qualified_name,
                "path": normalized_path,
                "definition_type": _normalize_text(artifact.get("definition_type")),
                "defined_line": artifact.get("defined_line_number"),
                "end_line": artifact.get("end_line_number"),
                "event_count": artifact.get("event_count_in_trace_a"),
                "call_count": artifact.get("call_count_in_trace_a"),
                "max_call_depth": artifact.get("max_call_depth"),
            }
        )

    page = filtered_artifacts[start : start + limit]
    result = {
        "instance_id": instance_id,
        "trace_summary": {
            "unique_function_count": artifact_data.get("comparison", {}).get(
                "unique_function_count",
                len(raw_artifacts),
            ),
            "returned_count": len(page),
            "filtered_count": len(filtered_artifacts),
        },
        "applied_filters": {
            "artifact_type": artifact_type,
            "path_contains": path_contains,
            "name_contains": name_contains,
            "start": start,
            "limit": limit,
        },
        "artifacts": page,
        "has_more": start + limit < len(filtered_artifacts),
        "next_start": start + len(page),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def _compact_symbol(symbol_id: str, symbol_info: list, file_map: dict) -> dict:
    file_id = symbol_info[3] if len(symbol_info) > 3 else ""
    raw_path = file_map.get(file_id, "")
    return {
        "symbol_id": symbol_id,
        "kind": symbol_info[0] if len(symbol_info) > 0 else "",
        "symbol": symbol_info[1] if len(symbol_info) > 1 else "",
        "scope": symbol_info[2] if len(symbol_info) > 2 else "",
        "path": _normalize_repo_path(raw_path),
        "line": symbol_info[4] if len(symbol_info) > 4 else None,
    }


def _matches_symbol_filters(
    symbol: dict,
    symbol_contains: str | None,
    scope_contains: str | None,
    path_contains: str | None,
) -> bool:
    return (
        _contains_casefold(_normalize_text(symbol.get("symbol")), symbol_contains)
        and _contains_casefold(_normalize_text(symbol.get("scope")), scope_contains)
        and _contains_casefold(_normalize_text(symbol.get("path")), path_contains)
    )


def _compress_symbol_path(path_symbol_ids: list[str], compact_symbols: dict[str, dict]) -> str:
    parts = []
    for symbol_id in path_symbol_ids:
        symbol = compact_symbols.get(symbol_id)
        if not symbol:
            continue
        symbol_name = symbol["symbol"]
        scope = symbol["scope"]
        path = symbol["path"]
        if scope and scope != "<module>":
            parts.append(f"{path}:{scope}.{symbol_name}")
        elif scope:
            parts.append(f"{path}:{scope}.{symbol_name}")
        else:
            parts.append(f"{path}:{symbol_name}")
    return " -> ".join(parts)


def get_trace_data_dependencies(
    symbol_contains: str | None = None,
    scope_contains: str | None = None,
    path_contains: str | None = None,
    edge_type: str | None = None,
    start: int = 0,
    limit: int = 12,
    path_limit: int = 4,
):
    """Return a compact raw slice of DaLIC data-dependency evidence for the current instance.

    This tool reads `unique_dep_tree.json` from `DaLIC/vt_results/<instance_id>/`.
    It does not rank or recommend results. It only applies exact filters and pagination,
    then returns a compact JSON summary.

    Tips:
    - Use it when you need a precise dependency slice around a traced symbol, scope, or file path.
    - You can call it again at any time during analysis with narrower filters such as `scope_contains`, `symbol_contains`, or `edge_type`.
    """
    instance_id = _require_current_instance_id()
    start = _clamp_int(start, default=0, maximum=50_000)
    limit = _clamp_int(limit, default=12, maximum=MAX_DEP_LIMIT)
    path_limit = _clamp_int(path_limit, default=4, maximum=MAX_PATH_LIMIT)

    dep_tree = _load_unique_dep_tree(instance_id)
    files = dep_tree.get("files", {})
    symbols = dep_tree.get("symbols", {})
    edges = dep_tree.get("edges", [])
    paths = dep_tree.get("paths", {})

    compact_symbols = {
        symbol_id: _compact_symbol(symbol_id, symbol_info, files)
        for symbol_id, symbol_info in symbols.items()
    }

    matched_symbol_ids = {
        symbol_id
        for symbol_id, compact_symbol in compact_symbols.items()
        if _matches_symbol_filters(
            compact_symbol,
            symbol_contains=symbol_contains,
            scope_contains=scope_contains,
            path_contains=path_contains,
        )
    }

    filtered_edges = []
    for edge in edges:
        if len(edge) < 3:
            continue
        src_id, dst_id, cur_edge_type = edge[:3]
        if edge_type and cur_edge_type != edge_type:
            continue
        if matched_symbol_ids and src_id not in matched_symbol_ids and dst_id not in matched_symbol_ids:
            continue
        filtered_edges.append(edge)

    page_edges = filtered_edges[start : start + limit]
    page_symbol_ids = set()
    compact_edges = []
    for edge in page_edges:
        src_id, dst_id, cur_edge_type = edge[:3]
        if src_id not in compact_symbols or dst_id not in compact_symbols:
            continue
        page_symbol_ids.add(src_id)
        page_symbol_ids.add(dst_id)
        compact_edges.append(
            {
                "src": {
                    key: value
                    for key, value in compact_symbols[src_id].items()
                    if key != "symbol_id"
                },
                "dst": {
                    key: value
                    for key, value in compact_symbols[dst_id].items()
                    if key != "symbol_id"
                },
                "edge_type": cur_edge_type,
            }
        )

    if matched_symbol_ids and not page_symbol_ids:
        for symbol_id in list(matched_symbol_ids)[: min(limit, 8)]:
            page_symbol_ids.add(symbol_id)

    compact_matched_symbols = [
        {
            key: value
            for key, value in compact_symbols[symbol_id].items()
            if key != "symbol_id"
        }
        for symbol_id in compact_symbols
        if symbol_id in page_symbol_ids
    ][:MAX_SYMBOL_SUMMARY]

    compact_paths = []
    if page_symbol_ids:
        for sink_symbol_id, path_groups in paths.items():
            if len(compact_paths) >= path_limit:
                break
            for path_symbol_ids in path_groups:
                if not any(symbol_id in page_symbol_ids for symbol_id in path_symbol_ids):
                    continue
                compressed_path = _compress_symbol_path(path_symbol_ids, compact_symbols)
                if compressed_path and compressed_path not in compact_paths:
                    compact_paths.append(compressed_path)
                break
            if len(compact_paths) >= path_limit:
                break

    result = {
        "instance_id": instance_id,
        "dep_summary": {
            "retained_file_count": dep_tree.get("filter_summary", {}).get("retained_file_count"),
            "retained_symbol_count": dep_tree.get("filter_summary", {}).get("retained_symbol_count"),
            "retained_edge_count": dep_tree.get("filter_summary", {}).get("retained_edge_count"),
            "returned_edge_count": len(compact_edges),
            "filtered_edge_count": len(filtered_edges),
        },
        "applied_filters": {
            "symbol_contains": symbol_contains,
            "scope_contains": scope_contains,
            "path_contains": path_contains,
            "edge_type": edge_type,
            "start": start,
            "limit": limit,
            "path_limit": path_limit,
        },
        "matched_symbols": compact_matched_symbols,
        "edges": compact_edges,
        "paths": compact_paths,
        "has_more": start + limit < len(filtered_edges),
        "next_start": start + len(page_edges),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


__all__ = ["get_trace_artifacts", "get_trace_data_dependencies"]
