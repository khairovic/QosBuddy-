"""QoSBuddy MLOps pipeline.

Replaces the manual notebook-and-copy workflow with scheduled scoring jobs:
the bridge writes raw rows into data/live_log.jsonl, the API runs the jobs
periodically, and they refresh the *_live.csv files the dashboard merges with
the static historical CSVs.

Public surface:
  from mlops import runner          # singleton MLOpsRunner
  from mlops.jobs import register_default_jobs
"""

from .runner import runner, MLOpsRunner
from .jobs import register_default_jobs

__all__ = ["runner", "MLOpsRunner", "register_default_jobs"]
