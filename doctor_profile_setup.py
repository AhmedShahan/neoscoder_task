"""Top-level launcher for the doctor profile setup CLI.

This mirrors how `workflow_execution.py` is runnable from the repository root.
Run with:

    python3 doctor_profile_setup.py

Which ensures the project root is on sys.path so `from src...` imports work.
"""
from src.workflow.doctor_profile_setup import main


if __name__ == "__main__":
    main()
