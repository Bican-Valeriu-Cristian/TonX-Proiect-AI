import os
import json
from datetime import datetime
from typing import Any, Dict, Optional, List

class MetricsLogger:
    def __init__(self, base_dir: str = "metrics"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _task_dir(self, task: str) -> str:
        d = os.path.join(self.base_dir, task)
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        return d

    def new_run_id(self) -> str:
        # ex: 2026-01-15_21-37-05
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def list_runs(self, task: str) -> List[str]:
        d = self._task_dir(task)
        runs_dir = os.path.join(d, "runs")
        if not os.path.exists(runs_dir):
            return []
        runs = []
        for fn in os.listdir(runs_dir):
            if fn.endswith(".json"):
                runs.append(fn.replace(".json", ""))
        runs.sort(reverse=True)
        return runs

    def save_metrics(self, task: str, metrics: Dict[str, Any], run_id: Optional[str] = None) -> str:
        """
        Salvează versionat + menține compatibilitatea cu vechiul fișier:
          - metrics/<task>/runs/<run_id>.json
          - metrics/<task>_metrics.json  (latest)
          - metrics/<task>/latest.json   (latest)
        Returnează run_id.
        """
        d = self._task_dir(task)
        run_id = run_id or self.new_run_id()

        payload = dict(metrics)
        payload["run_id"] = run_id
        payload["last_updated"] = datetime.now().isoformat(timespec="seconds")

        run_path = os.path.join(d, "runs", f"{run_id}.json")
        with open(run_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # latest.json (în folderul task-ului)
        latest_path = os.path.join(d, "latest.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # Backward-compat: vechiul format folosit de UI-ul tău curent
        legacy_latest_path = os.path.join(self.base_dir, f"{task}_metrics.json")
        with open(legacy_latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return run_id

    def load_metrics(self, task: str, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Dacă run_id e None:
          - încearcă legacy: metrics/<task>_metrics.json
          - apoi metrics/<task>/latest.json
        Dacă run_id e dat:
          - metrics/<task>/runs/<run_id>.json
        """
        d = self._task_dir(task)

        if run_id:
            run_path = os.path.join(d, "runs", f"{run_id}.json")
            if not os.path.exists(run_path):
                return None
            with open(run_path, "r", encoding="utf-8") as f:
                return json.load(f)

        legacy_latest_path = os.path.join(self.base_dir, f"{task}_metrics.json")
        if os.path.exists(legacy_latest_path):
            with open(legacy_latest_path, "r", encoding="utf-8") as f:
                return json.load(f)

        latest_path = os.path.join(d, "latest.json")
        if os.path.exists(latest_path):
            with open(latest_path, "r", encoding="utf-8") as f:
                return json.load(f)

        return None
