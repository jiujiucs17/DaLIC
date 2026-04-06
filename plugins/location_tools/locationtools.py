from inspect import signature

from plugins.location_tools import repo_ops, retriever
from DaLIC import raw_tools
from plugins.location_tools.utils.dependency import import_functions

# import_functions(
#     module=retriever, function_names=retriever.__all__, target_globals=globals()
# )

import_functions(
    module=repo_ops, function_names=repo_ops.__all__, target_globals=globals()
)
import_functions(
    module=raw_tools, function_names=raw_tools.__all__, target_globals=globals()
)
__all__ = repo_ops.__all__ + raw_tools.__all__  # + retriever.__all__

GRAPH_TOOL_NAMES = {
    "explore_graph_structure",
    "explore_tree_structure",
}


def build_documentation(include_graph: bool = True) -> str:
    documentation = ''
    function_names = __all__
    if not include_graph:
        function_names = [
            func_name for func_name in function_names
            if func_name not in GRAPH_TOOL_NAMES
        ]

    for func_name in function_names:
        func = globals()[func_name]

        cur_doc = func.__doc__
        # remove indentation from docstring and extra empty lines
        cur_doc = '\n'.join(filter(None, map(lambda x: x.strip(), cur_doc.split('\n'))))
        # now add a consistent 4 indentation
        cur_doc = '\n'.join(map(lambda x: ' ' * 4 + x, cur_doc.split('\n')))

        fn_signature = f'{func.__name__}' + str(signature(func))
        documentation += f'{fn_signature}:\n{cur_doc}\n\n'

    return documentation


DOCUMENTATION = build_documentation()
