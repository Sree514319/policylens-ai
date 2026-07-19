"""Helper invoked as a standalone subprocess by
`test_entrypoint_scripts_execute_cleanly_in_a_genuinely_isolated_subprocess`
in test_app_smoke.py -- never imported or collected by pytest itself.

Runs `AppTest.from_file(<path>)` against a single target script and reports
the outcome on stdout. Deliberately run as its own separate Python process
(not called in-process from a test function) so that pytest.ini's
`pythonpath = backend frontend` setting -- which only patches `sys.path` for
the pytest process itself, via pytest's own startup hook, and is never
inherited by a subprocess -- cannot mask an import bug that only manifests
when `frontend/` is genuinely absent from `sys.path`. That gap is exactly
how the original `ModuleNotFoundError: No module named 'streamlit_app'`
production bug passed every in-process AppTest-based test while still
failing under a real `streamlit run`.
"""

import sys

from streamlit.testing.v1 import AppTest

target = sys.argv[1]
at = AppTest.from_file(target)
at.run(timeout=30)

if at.exception:
    for exc in at.exception:
        print("EXCEPTION:", getattr(exc, "value", exc))
        stack_trace = getattr(exc, "stack_trace", None)
        if stack_trace:
            print("\n".join(stack_trace))
    sys.exit(1)

print("ISOLATED_APPTEST_OK")
sys.exit(0)
