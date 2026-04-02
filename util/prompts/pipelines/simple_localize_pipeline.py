SEARCH_LOC_TASK_INSTRUCTION="""
# Task:
You will be provided with a GitHub issue description. Your objective is to localize the specific files, classes or functions, and lines of code that are relevant to the solution for the issue. More specifically, these issues are all "feature request" issues, which means that you need to find the locations the feature described in the issue.
1. Analyze the issue: Understand the requested feature described in the issue and identify what related to it.
2. Extract the Necessary Search Parameters from the issue and call retrieval-based functions.
3. Locate the specific files, functions, methods, or lines of code that are relevant to the feature.
"""


OUTPUT_FORMAT_LOC="""
# Output Format for Search Results:
Your final output should list the locations requiring modification, wrapped with triple backticks ```
Each location should include the file path, class name (if applicable), function name, or line numbers, ordered by importance.

IMPORTANT FORMAT RULES:
- Output ONLY the location block, followed by `<finish></finish>`.
- Do NOT include explanations, bullets, headings, numbering, summaries, or notes before or after the location block.
- Each file path MUST appear on its own line and end with `.py`.
- Do NOT write `file.py:QualifiedName` on a single line in the final answer.
- After a file path line, use separate `line:`, `class:`, and `function:` lines as needed.

## Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
function: my_function1

full_path2/file2.py
line: 76
function: MyClass2.my_function2

full_path3/file3.py
line: 24
line: 156
function: my_function3
```

Return just the location(s)
"""

FINAL_OUTPUT_REWRITE_REMINDER = """
Your last answer did not follow the required final output format.

Rewrite your final answer now using this exact structure:

```
path/to/file1.py
line: 10-20
class: MyClass
function: my_function

path/to/file2.py
line: 42
function: another_function
```
<finish></finish>

Rules:
- Output ONLY one fenced code block, then `<finish></finish>`
- Each file path must be on its own line and end with `.py`
- Do NOT use `file.py:QualifiedName` on one line
- Do NOT include explanations, headings, bullets, numbering, or any prose
- Return just the location(s)
"""


FAKE_USER_MSG_FOR_LOC = (
        'Verify if the found locations contain all the necessary information to address the issue, and check for any relevant references in other parts of the codebase that may not have appeared in the search results. '
        'If not, continue searching for additional locations related to the issue.\n'
        'Verify that you have carefully analyzed the impact of the found locations on the repository, especially their dependencies. '
        'If you think you have solved the task, rewrite your final answer into the required location-only format, output ONLY the location block, and then use the following command to finish: <finish></finish>.\n'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
)
