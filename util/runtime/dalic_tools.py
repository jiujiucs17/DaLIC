from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk


_TRACE_ARTIFACTS_DESCRIPTION = """
Read a compact raw slice of DaLIC execution-trace artifacts for the current instance.
This tool reads `DaLIC/vt_results/<instance_id>/unique_artifacts.json` and returns a short JSON summary.
It does not rank or recommend results. It only applies exact filters and pagination.

Usage suggestions:
- Call this once near the start of analyzing a new data point to inspect execution-trace evidence.
- Call it again later with narrower `path_contains` or `name_contains` filters whenever you need a more precise slice.
"""


TraceArtifactsTool = ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="get_trace_artifacts",
        description=_TRACE_ARTIFACTS_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "artifact_type": {
                    "type": "string",
                    "enum": ["all", "function", "method", "class"],
                    "default": "all",
                    "description": "Optional exact filter for artifact definition type.",
                },
                "path_contains": {
                    "type": ["string", "null"],
                    "description": "Optional case-insensitive substring filter applied to the compact file path.",
                    "default": None,
                },
                "name_contains": {
                    "type": ["string", "null"],
                    "description": "Optional case-insensitive substring filter applied to the qualified name.",
                    "default": None,
                },
                "start": {
                    "type": "integer",
                    "default": 0,
                    "description": "Zero-based pagination offset.",
                },
                "limit": {
                    "type": "integer",
                    "default": 8,
                    "description": "Maximum number of artifacts to return in this slice.",
                },
            },
            "required": [],
        },
    ),
)


_TRACE_DEPS_DESCRIPTION = """
Read a compact raw slice of DaLIC data-dependency evidence for the current instance.
This tool reads `DaLIC/vt_results/<instance_id>/unique_dep_tree.json` and returns a short JSON summary.
It does not rank or recommend results. It only applies exact filters and pagination.

Usage suggestions:
- Use it whenever you need a focused dependency slice around a traced symbol, scope, or file path.
- During analysis, you may call it again with narrower `symbol_contains`, `scope_contains`, `path_contains`, or `edge_type` filters.
"""


TraceDataDependenciesTool = ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="get_trace_data_dependencies",
        description=_TRACE_DEPS_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "symbol_contains": {
                    "type": ["string", "null"],
                    "description": "Optional case-insensitive substring filter applied to symbol names.",
                    "default": None,
                },
                "scope_contains": {
                    "type": ["string", "null"],
                    "description": "Optional case-insensitive substring filter applied to symbol scopes.",
                    "default": None,
                },
                "path_contains": {
                    "type": ["string", "null"],
                    "description": "Optional case-insensitive substring filter applied to compact file paths.",
                    "default": None,
                },
                "edge_type": {
                    "type": ["string", "null"],
                    "description": "Optional exact filter for dependency edge type such as `data` or `ret`.",
                    "default": None,
                },
                "start": {
                    "type": "integer",
                    "default": 0,
                    "description": "Zero-based pagination offset for matched edges.",
                },
                "limit": {
                    "type": "integer",
                    "default": 12,
                    "description": "Maximum number of matched edges to return in this slice.",
                },
                "path_limit": {
                    "type": "integer",
                    "default": 4,
                    "description": "Maximum number of compact dependency paths to include.",
                },
            },
            "required": [],
        },
    ),
)
