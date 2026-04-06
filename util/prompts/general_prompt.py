PR_TEMPLATE = """
--- BEGIN PROBLEM STATEMENT ---
Title: {title}

{description}
--- END PROBLEM STATEMENT ---

"""


SYSTEM_PROMPT="""You're an experienced software tester and static analysis expert. 
Given the problem offered by the user, please perform a thorough static analysis and to localize the bug in this repository using the available tools.

Focus on:
- Identifying any deviations, potential errors, or unexpected behavior that could contribute to the issue.
- Considering how dynamic binding, late resolution, or other runtime behavior may influence the code's behavior.
- Highlighting possible root causes or key areas for further inspection.
"""
