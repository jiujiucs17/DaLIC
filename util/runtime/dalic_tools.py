from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk


_TRACE_ARTIFACTS_DESCRIPTION = """
Read a compact raw slice of DaLIC execution-trace artifacts for the current instance.
This tool reads '<instance_id>/unique_artifacts.json' and returns a short JSON summary.
It does not rank or recommend results. It only applies exact filters and pagination.

Note:
'<instance_id>/unique_artifacts.json' is generated through a Software Reconnaissance-style analysis, which involves:
- executing two Python scripts describing the same process with (A.py) and without (B.py) the feature enabled.
- tracing their execution, obtain trace A and trace B as the execution traces of A.py and B.py, respectively.
- then subtracting the overlapping parts from trace A and save the rest in '<instance_id>/unique_artifacts.json'. Therefore it contains the non-overlapping artifacts that only appear in A.py, which are likely to be related to the feature and thus to the issue.

Args:
- `artifact_type`: Optional exact filter for artifact definition type. It can be `function`, `method`, or `class`. 
    If not specified, it defaults to `all` and does not filter by artifact type.
- `path_contains`: Optional case-insensitive substring filter applied to the compact file path. For example, if you are interested in artifacts defined in files under `src/utils`, you can set `path_contains` to `src/utils`. 
    If not specified, it defaults to `None` and does not filter by file path.
- `name_contains`: Optional case-insensitive substring filter applied to the qualified name. For example, if you are interested in artifacts with `DataLoader` in their qualified names, you can set `name_contains` to `DataLoader`. 
    If not specified, it defaults to `None` and does not filter by qualified name.
- `start`: Zero-based pagination offset. It defaults to 0, meaning that the first page starts from the first matched artifact.
- `limit`: Maximum number of artifacts to return in this slice. It defaults to 8, which is a reasonable number for you to inspect at a time. You can adjust it based on your preference and the typical number of matched artifacts.

Returns:
A JSON string encoding an object with the following top-level fields:
- `instance_id`: The current DaLIC instance id.
- `trace_summary`: Summary counts, including `unique_function_count`, `returned_count`, and `filtered_count`.
- `applied_filters`: The effective filter and pagination values used for this call.
- `artifacts`: A JSON array of matched artifacts.
- `has_more`: Whether more matched artifacts are available after this slice.
- `next_start`: The pagination offset to use for the next slice.

Each artifact in `artifacts` is a JSON object with the following fields:
- `qualified_name`: The qualified name of the artifact.
- `path`: The compact file path where the artifact is defined.
- `definition_type`: The raw definition type from DaLIC, such as `function` or `class`.
- `defined_line`: The line number where the artifact is defined.
- `end_line`: The ending line number of the artifact, if available.
- `event_count`: The number of matching execution events in trace A.
- `call_count`: The number of calls observed in trace A.
- `max_call_depth`: The maximum observed call depth for this artifact.

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
This tool reads '<instance_id>/unique_dep_tree.json' and returns a short JSON summary.
It does not rank or recommend results. It only applies filters and pagination.

Note:
- '<instance_id>/unique_dep_tree.json' is generated by analyzing the data dependencies of the unique artifacts in '<instance_id>/unique_artifacts.json'.  
- It contains the dataflows associated with the unique artifacts, which can help you understand how the unique artifacts are related to each other and how they are related to other parts of the codebase.

Args:
- `symbol_contains`: Optional case-insensitive substring filter applied to symbol names. 
    For example, if you are interested in dependencies related to symbols with `DataLoader` in their names, you can set `symbol_contains` to `DataLoader`. 
    Together with `scope_contains` and `path_contains`, this first selects a set of matching symbols; an edge is retained when either endpoint belongs to that set.
    If not specified, it defaults to `None`
- `scope_contains`: Optional case-insensitive substring filter applied to symbol scopes. 
    The scope is represented as a dot-separated string of nested scopes, such as `src.utils.data_loader.DataLoader.__init__`. 
    You can set `scope_contains` to a substring of the scope to filter dependencies related to that scope. 
    For example, setting it to `DataLoader` would match the above scope and include its dependencies.
    Together with `symbol_contains` and `path_contains`, this contributes to the symbol-matching step described above.
    If not specified, it defaults to `None`
- `path_contains`: Optional case-insensitive substring filter applied to compact file paths.
    For example, if you are interested in dependencies related to files under `src/utils`, you can set `path_contains` to `src/utils`. 
    Together with `symbol_contains` and `scope_contains`, this contributes to the symbol-matching step described above.
    If not specified, it defaults to `None`
- `edge_type`: Optional exact filter for dependency edge type such as `data` or `ret`. If not specified, it defaults to `None` and does not filter by edge type.
- `start`: Zero-based pagination offset for matched edges. It defaults to 0, meaning that the first page starts from the first matched edge.
- `limit`: Maximum number of matched edges to return in this slice. It defaults to 12, which is a reasonable number for you to inspect at a time. 
    You can adjust it based on your preference and the typical number of matched edges.
- `path_limit`: Maximum number of compact dependency paths to include. A dependency path is a sequence of dependency edges connecting two symbols. 
    If there are more matched paths than this limit, only the first few unique compressed paths are returned.

Important behavior:
- If at least one symbol matches the combined `symbol_contains` / `scope_contains` / `path_contains` filters, only edges whose source or destination is in that matched symbol set are retained.
- If no symbol matches those filters, edge filtering falls back to `edge_type` plus pagination only; it does not return an empty edge list.

Returns:
A JSON string encoding an object with the following top-level fields:
- `instance_id`: The current DaLIC instance id.
- `dep_summary`: Summary counts, including retained file / symbol / edge counts and the returned / filtered edge counts.
- `applied_filters`: The effective filter and pagination values used for this call.
- `matched_symbols`: A compact JSON array of symbols related to the current page of results.
- `edges`: A JSON array of matched dependency edges.
- `paths`: A JSON array of compressed dependency-path strings related to the current page.
- `has_more`: Whether more matched edges are available after this slice.
- `next_start`: The pagination offset to use for the next slice.

Each symbol in `matched_symbols` has these fields:
- `kind`: The raw symbol kind from DaLIC, such as `func`, `class`, or similar labels.
- `symbol`: The symbol name.
- `scope`: The raw scope string associated with the symbol.
- `path`: The compact file path where the symbol is defined.
- `line`: The line number where the symbol is defined.

Each edge in `edges` is a JSON object with the following fields:
- `src`: A compact source-symbol object with fields `kind`, `symbol`, `scope`, `path`, and `line`.
- `dst`: A compact target-symbol object with fields `kind`, `symbol`, `scope`, `path`, and `line`.
- `edge_type`: The raw dependency edge type, such as `arg`, `data`, or `ret`.

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
