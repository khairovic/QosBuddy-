"""MLOpsRunner — register jobs, run them on demand or on a periodic timer.

Design:
- Jobs are pure callables that produce dict metrics. They write outputs
  atomically (write `.tmp`, then `os.replace`) so a partial file never
  becomes visible to the dashboard.
- One job runs at a time per name (in-process lock). Parallel jobs across
  names are allowed.
- State (last_run, status, metrics) persists to data/.mlops_state.json so the
  dashboard can show last-run timestamps after API restarts.
- Scheduler is `threading.Timer` based (no extra dependency).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

log = logging.getLogger("qosbuddy.mlops")

DATA_DIR = Path(os.getenv("QOSBUDDY_DATA_DIR", "./data"))
STATE_PATH = DATA_DIR / ".mlops_state.json"


@dataclass
class JobRecord:
    name: str
    last_run:        Optional[str] = None     # ISO timestamp
    last_status:     str           = "never"  # never | running | ok | fail
    last_duration_s: float         = 0.0
    last_error:      Optional[str] = None
    last_metrics:    Dict[str, Any] = field(default_factory=dict)
    runs:            int           = 0
    fails:           int           = 0


class MLOpsRunner:
    """Singleton-style runner. Use `mlops.runner` everywhere."""

    def __init__(self) -> None:
        self._jobs:    Dict[str, Callable[[], Dict[str, Any]]] = {}
        self._records: Dict[str, JobRecord] = {}
        self._save_lock  = threading.Lock()
        self._run_locks: Dict[str, threading.Lock] = {}
        self._timer:   Optional[threading.Timer] = None
        # RLock — start_scheduler holds the lock then calls _tick which
        # also takes the lock; a plain Lock would deadlock the API at boot.
        self._timer_lock = threading.RLock()
        self._interval_s = int(os.getenv("QOSBUDDY_PIPELINE_INTERVAL", "300"))
        self._auto_start = os.getenv("QOSBUDDY_PIPELINE_AUTO", "true").lower() in {"1","true","yes"}
        self._load_state()

    # ── Registration ────────────────────────────────────────────────────
    def register(self, name: str, fn: Callable[[], Dict[str, Any]]) -> None:
        self._jobs[name] = fn
        self._run_locks.setdefault(name, threading.Lock())
        if name not in self._records:
            self._records[name] = JobRecord(name=name)
        log.info(f"mlops: registered job '{name}'")

    # ── Execution ───────────────────────────────────────────────────────
    def run_one(self, name: str) -> Dict[str, Any]:
        if name not in self._jobs:
            return {"status": "unknown_job", "error": f"no job named '{name}'"}
        lock = self._run_locks[name]
        if not lock.acquire(blocking=False):
            return {"status": "already_running"}
        try:
            return self._execute(name)
        finally:
            lock.release()

    def _execute(self, name: str) -> Dict[str, Any]:
        rec = self._records.setdefault(name, JobRecord(name=name))
        rec.last_status = "running"
        self._save_state()
        t0 = time.time()
        try:
            metrics = self._jobs[name]() or {}
            if not isinstance(metrics, dict):
                metrics = {"result": str(metrics)}
            rec.last_status  = "ok"
            rec.last_metrics = metrics
            rec.last_error   = None
        except Exception as e:
            log.exception(f"mlops: job '{name}' failed")
            rec.last_status  = "fail"
            rec.last_error   = str(e)[:300]
            rec.fails       += 1
            metrics = {}
        rec.last_duration_s = round(time.time() - t0, 3)
        rec.last_run        = datetime.now().isoformat(timespec="seconds")
        rec.runs           += 1
        self._save_state()
        return {
            "status":     rec.last_status,
            "duration_s": rec.last_duration_s,
            "error":      rec.last_error,
            **metrics,
        }

    def run_all(self) -> Dict[str, Dict[str, Any]]:
        return {name: self.run_one(name) for name in list(self._jobs)}

    # ── Scheduler ───────────────────────────────────────────────────────
    def start_scheduler(self, interval_s: Optional[int] = None) -> None:
        if interval_s and interval_s > 0:
            self._interval_s = int(interval_s)
        with self._timer_lock:
            self._cancel_timer_locked()
            self._tick(initial_delay=True)
        log.info(f"mlops: scheduler started, interval={self._interval_s}s")

    def stop_scheduler(self) -> None:
        with self._timer_lock:
            self._cancel_timer_locked()
        log.info("mlops: scheduler stopped")

    def _tick(self, initial_delay: bool = False) -> None:
        if not initial_delay:
            try:
                self.run_all()
            except Exception:
                log.exception("mlops: scheduler tick failed")
        with self._timer_lock:
            self._timer = threading.Timer(self._interval_s, self._tick)
            self._timer.daemon = True
            self._timer.start()

    def _cancel_timer_locked(self) -> None:
        if self._timer:
            try: self._timer.cancel()
            except Exception: pass
            self._timer = None

    # ── Status / state ──────────────────────────────────────────────────
    def status(self) -> Dict[str, Any]:
        return {
            "interval_s": self._interval_s,
            "scheduler_running": self._timer is not None,
            "auto_start_on_boot": self._auto_start,
            "jobs": {n: asdict(r) for n, r in self._records.items()},
        }

    def _save_state(self) -> None:
        with self._save_lock:
            try:
                STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "interval_s": self._interval_s,
                    "jobs": {n: asdict(r) for n, r in self._records.items()},
                }
                tmp = STATE_PATH.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                os.replace(tmp, STATE_PATH)
            except Exception:
                log.warning("mlops: could not persist state", exc_info=True)

    def _load_state(self) -> None:
        if not STATE_PATH.exists():
            return
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            for name, rec in (data.get("jobs") or {}).items():
                # JobRecord may have grown fields — pass only the known ones.
                allowed = {f for f in JobRecord.__dataclass_fields__}
                clean   = {k: v for k, v in rec.items() if k in allowed}
                self._records[name] = JobRecord(**clean)
            if "interval_s" in data:
                self._interval_s = int(data["interval_s"])
        except Exception:
            log.warning("mlops: could not load prior state", exc_info=True)

    @property
    def auto_start_on_boot(self) -> bool:
        return self._auto_start


# Singleton — every importer gets the same instance.
runner = MLOpsRunner()
