"""Shared, side-effect-free helpers for Cascade qualification runners."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import threading

try:
    import torch
except ModuleNotFoundError as error:  # Plan tools do not require Torch.
    if error.name != "torch":
        raise
    torch = None


PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED_WITH_REASON"
BLOCKED = "BLOCKED"
BLOCKED_NOT_EXCLUSIVE = "BLOCKED_NOT_EXCLUSIVE"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_revision(root):
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except BaseException:
        return None


def _run_nvidia_smi(arguments):
    try:
        completed = subprocess.run(
            ["nvidia-smi"] + list(arguments), check=True,
            capture_output=True, text=True, timeout=5,
        )
        return completed.stdout, None
    except BaseException as error:
        return "", "{}: {}".format(type(error).__name__, error)


def _gpu_inventory():
    output, error = _run_nvidia_smi(
        [
            "--query-gpu=index,uuid,name,driver_version,memory.total,"
            "memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if error:
        return [], error
    result = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",", 6)]
        if len(fields) != 7:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        def optional_integer(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        total_memory_mib = optional_integer(fields[4])
        used_memory_mib = optional_integer(fields[5])
        utilization_gpu_percent = optional_integer(fields[6])
        result.append(
            {
                "index": index,
                "uuid": fields[1],
                "name": fields[2],
                "driver_version": fields[3],
                "total_memory_mib": total_memory_mib,
                "used_memory_mib": used_memory_mib,
                "utilization_gpu_percent": utilization_gpu_percent,
            }
        )
    return result, None


def _compute_processes():
    output, error = _run_nvidia_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    if error:
        return [], error
    result = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",", 3)]
        if len(fields) != 4:
            continue
        try:
            pid = int(fields[1])
            used_memory = int(fields[2])
        except ValueError:
            continue
        result.append(
            {
                "gpu_uuid": fields[0],
                "pid": pid,
                "used_memory_mib": used_memory,
                "process_name": os.path.basename(fields[3]),
                "process_path_redacted": os.path.basename(fields[3]) != fields[3],
                "is_current_process": pid == os.getpid(),
            }
        )
    return result, None


def cuda_device_index(device):
    if torch is None:
        value = str(device)
        if not value.startswith("cuda"):
            raise ValueError("device must select CUDA")
        if ":" not in value:
            return 0
        return int(value.split(":", 1)[1])
    value = torch.device(device)
    if value.type != "cuda":
        raise ValueError("device must select CUDA")
    return 0 if value.index is None else int(value.index)


def resolve_physical_gpu(device, inventory):
    """Map a logical Torch device through ``CUDA_VISIBLE_DEVICES``."""

    logical = cuda_device_index(device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        matches = [item for item in inventory if item["index"] == logical]
    else:
        entries = [item.strip() for item in visible.split(",") if item.strip()]
        if logical >= len(entries):
            return None, "logical CUDA device is outside CUDA_VISIBLE_DEVICES"
        selector = entries[logical]
        if selector.isdigit():
            matches = [
                item for item in inventory if item["index"] == int(selector)
            ]
        else:
            matches = [
                item for item in inventory
                if item["uuid"] == selector or item["uuid"].startswith(selector)
            ]
    if len(matches) != 1:
        return None, "cannot uniquely resolve selected physical GPU"
    return matches[0], None


def capture_cuda_environment(root, device="cuda:0"):
    """Capture the selected physical GPU and external compute processes.

    This is a point-in-time admission check, not an exclusive reservation.
    Runners must keep that distinction explicit in generated reports.
    """

    inventory, inventory_error = _gpu_inventory()
    processes, process_error = _compute_processes()
    selected, resolution_error = resolve_physical_gpu(device, inventory)
    selected_processes = []
    if selected is not None:
        selected_processes = [
            item for item in processes if item["gpu_uuid"] == selected["uuid"]
        ]
    external = [item for item in selected_processes if not item["is_current_process"]]
    slurm = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "step_id": os.environ.get("SLURM_STEP_ID"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "step_gpus": os.environ.get("SLURM_STEP_GPUS"),
        "tres_per_node": os.environ.get("SLURM_TRES_PER_NODE"),
        "job_gres": os.environ.get("SLURM_JOB_GRES"),
        "account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
    }
    gpu_allocation = any(
        slurm.get(name)
        for name in ("job_gpus", "step_gpus", "tres_per_node", "job_gres")
    )
    reservation_evidence_present = bool(slurm["job_id"] and gpu_allocation)
    result = {
        "captured_at": utc_now(),
        "git_revision": git_revision(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": None if torch is None else torch.__version__,
        "torch_cuda": None if torch is None else torch.version.cuda,
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "cuda_device_count": (
            0 if torch is None else int(torch.cuda.device_count())
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requested_device": str(device),
        "selected_physical_gpu": selected,
        "selected_gpu_compute_processes": selected_processes,
        "selected_gpu_external_compute_processes": len(external),
        "exclusive_snapshot": bool(selected is not None and not external),
        "reservation_evidence": {"scheduler": "slurm", **slurm},
        "reservation_evidence_present": reservation_evidence_present,
        "nvidia_smi_errors": [
            item for item in (inventory_error, process_error, resolution_error)
            if item
        ],
    }
    if torch is not None and torch.cuda.is_available():
        try:
            index = cuda_device_index(device)
            result["torch_device"] = {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(
                    torch.cuda.get_device_properties(index).total_memory
                ),
            }
        except BaseException as error:
            result["torch_device_error"] = "{}: {}".format(
                type(error).__name__, error
            )
    return result


def qualification_admission(
    environment, allow_shared_smoke=False, require_reservation=False
):
    """Return ``(admitted, status, reason)`` for a selected CUDA GPU."""

    if not environment.get("cuda_available"):
        return False, SKIPPED, "torch.cuda.is_available() is false"
    if environment.get("selected_physical_gpu") is None:
        return False, BLOCKED, "selected physical GPU could not be resolved"
    external = int(environment.get("selected_gpu_external_compute_processes", 0))
    if external and not allow_shared_smoke:
        return (
            False,
            BLOCKED,
            "{}: selected GPU has {} external compute process(es)".format(
                BLOCKED_NOT_EXCLUSIVE, external
            ),
        )
    if require_reservation and not environment.get(
        "reservation_evidence_present", False
    ):
        return (
            False,
            BLOCKED,
            "BLOCKED_NO_RESERVATION: no scheduler-backed GPU allocation "
            "evidence is present",
        )
    return True, PASS, None


class GPUActivityMonitor:
    """Sample one physical GPU while a benchmark child is running.

    This is diagnostic interference detection, not a scheduler reservation.
    Only observations containing an external compute process are retained so
    report size stays bounded.
    """

    def __init__(self, gpu_uuid, *, interval_seconds=1.0):
        self.gpu_uuid = str(gpu_uuid)
        self.interval_seconds = max(0.1, float(interval_seconds))
        self._allowed_pids = {os.getpid()}
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._sample_count = 0
        self._external_observations = []
        self._errors = []
        self._peak_used_memory_mib = None
        self._peak_utilization_gpu_percent = None

    def allow_pid(self, pid):
        with self._lock:
            self._allowed_pids.add(int(pid))

    def _sample(self):
        inventory, inventory_error = _gpu_inventory()
        processes, process_error = _compute_processes()
        with self._lock:
            allowed = set(self._allowed_pids)
        gpu = next(
            (item for item in inventory if item["uuid"] == self.gpu_uuid),
            None,
        )
        selected = [
            item for item in processes if item["gpu_uuid"] == self.gpu_uuid
        ]
        external = [item for item in selected if item["pid"] not in allowed]
        with self._lock:
            self._sample_count += 1
            for error in (inventory_error, process_error):
                if error:
                    self._errors.append(error)
            if gpu is not None:
                used = gpu.get("used_memory_mib")
                utilization = gpu.get("utilization_gpu_percent")
                if used is not None:
                    self._peak_used_memory_mib = max(
                        int(used), self._peak_used_memory_mib or 0
                    )
                if utilization is not None:
                    self._peak_utilization_gpu_percent = max(
                        int(utilization),
                        self._peak_utilization_gpu_percent or 0,
                    )
            if external:
                self._external_observations.append(
                    {
                        "captured_at": utc_now(),
                        "processes": external,
                    }
                )

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def start(self):
        if self._thread is not None:
            raise RuntimeError("GPU activity monitor already started")
        self._thread = threading.Thread(
            target=self._run,
            name="cascade-gpu-activity-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2.0))
        with self._lock:
            return {
                "sample_count": self._sample_count,
                "external_activity_observed": bool(
                    self._external_observations
                ),
                "external_observations": list(self._external_observations),
                "peak_used_memory_mib": self._peak_used_memory_mib,
                "peak_utilization_gpu_percent": (
                    self._peak_utilization_gpu_percent
                ),
                "errors": list(dict.fromkeys(self._errors)),
                "reservation_provided": False,
            }


def write_report_bundle(output_dir, environment, cases, summary, markdown):
    """Write the common four-file qualification report bundle."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = summary.get("generated_at", utc_now())
    cases_document = {
        "schema_version": summary["schema_version"],
        "generated_at": generated_at,
        "mode": summary["mode"],
        "cases": list(cases),
    }
    documents = {
        "environment.json": environment,
        "cases.json": cases_document,
        "summary.json": summary,
    }
    for name, document in documents.items():
        (output_dir / name).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    return {name: str(output_dir / name) for name in documents} | {
        "report.md": str(output_dir / "report.md")
    }
