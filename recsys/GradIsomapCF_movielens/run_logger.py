import atexit
import datetime
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback


def _jsonable(o, _depth=0):
    if _depth > 50:
        return str(o)
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, int):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [_jsonable(v, _depth + 1) for v in o]
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return _jsonable(o.item(), _depth + 1)
        if isinstance(o, np.ndarray):
            if o.size <= 1_000_000:
                return _jsonable(o.tolist(), _depth + 1)
            return {"__ndarray__": True, "shape": list(o.shape), "dtype": str(o.dtype)}
    except ImportError:
        pass
    try:
        import torch
        if isinstance(o, torch.Tensor):
            if o.numel() <= 1_000_000:
                return _jsonable(o.detach().cpu().tolist(), _depth + 1)
            return {"__tensor__": True, "shape": list(o.shape), "dtype": str(o.dtype)}
        if isinstance(o, torch.device):
            return str(o)
    except ImportError:
        pass
    return str(o)


class _Tee:
    def __init__(self, orig, logger, tag):
        self._orig = orig
        self._logger = logger
        self._tag = tag

    def write(self, s):
        n = self._orig.write(s)
        try:
            self._logger._feed(self._tag, s)
        except Exception:
            pass
        return n

    def flush(self):
        self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


class RunLogger:
    _ORDER = ["status", "started_at", "finished_at", "duration_s", "config", "argv", "env",
              "data", "pure_ncf", "gi", "summary", "error"]

    def __init__(self, path, autosave_s=60.0):
        self.path = path
        self.autosave_s = autosave_s
        self.sections = {}
        self.console = []
        self.status = "not_started"
        self.error = None
        self._t0 = None
        self._started_at = None
        self._finished_at = None
        self._duration = None
        self._partial = {"out": "", "err": ""}
        self._last_save = 0.0
        self._saving = False
        self._warned = False
        self._orig_out = None
        self._orig_err = None
        self._tee_out = None
        self._tee_err = None
        self._orig_hook = None
        self._active = False

    def start(self, config=None):
        if self._active:
            return self
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        if os.path.exists(self.path):
            stamp = datetime.datetime.fromtimestamp(os.path.getmtime(self.path)).strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(self.path)
            try:
                os.replace(self.path, f"{base}.{stamp}{ext}")
            except OSError:
                pass

        self._t0 = time.time()
        self._started_at = datetime.datetime.now().isoformat(timespec="seconds")
        self.status = "running"
        self.sections["config"] = config or {}
        self.sections["argv"] = list(sys.argv)
        self.sections["env"] = self._collect_env()

        self._orig_out, self._orig_err = sys.stdout, sys.stderr
        self._tee_out = _Tee(sys.stdout, self, "out")
        self._tee_err = _Tee(sys.stderr, self, "err")
        sys.stdout, sys.stderr = self._tee_out, self._tee_err
        self._orig_hook = sys.excepthook
        sys.excepthook = self._excepthook
        atexit.register(self._atexit)
        self._active = True
        self.save()
        print(f"[RunLog] лог прогона: {self.path}")
        return self

    def set(self, key, value, save=True):
        self.sections[key] = value
        if save:
            self.save()

    def finish(self, status="finished"):
        if not self._active:
            return
        for tag in ("out", "err"):
            if self._partial[tag]:
                self._append(tag, self._partial[tag])
                self._partial[tag] = ""
        if self.status == "running":
            self.status = status
        self._finished_at = datetime.datetime.now().isoformat(timespec="seconds")
        self._duration = round(time.time() - self._t0, 3)
        if sys.stdout is self._tee_out:
            sys.stdout = self._orig_out
        if sys.stderr is self._tee_err:
            sys.stderr = self._orig_err
        if sys.excepthook == self._excepthook:
            sys.excepthook = self._orig_hook
        self._active = False
        try:
            atexit.unregister(self._atexit)
        except Exception:
            pass
        self.save(force=True)
        print(f"[RunLog] сохранено: {self.path} (status={self.status})")

    def _excepthook(self, etype, value, tb):
        self.error = "".join(traceback.format_exception(etype, value, tb))
        self.status = "interrupted" if issubclass(etype, KeyboardInterrupt) else "failed"
        self._orig_hook(etype, value, tb)

    def _atexit(self):
        if self._active:
            self.finish(status=self.status if self.status != "running" else "exited_without_finish")

    def _feed(self, tag, s):
        buf = self._partial[tag] + s
        *lines, rest = buf.split("\n")
        self._partial[tag] = rest
        for line in lines:
            self._append(tag, line)
        if (not self._saving and self._t0 is not None
                and time.time() - self._last_save >= self.autosave_s):
            self.save()

    def _append(self, tag, line):
        line = line.rstrip("\r")
        if "\r" in line:
            line = line.split("\r")[-1]
        stamp = f"[{time.time() - self._t0:9.2f}s]"
        self.console.append(f"{stamp} {line}" if tag == "out" else f"{stamp} [stderr] {line}")

    @staticmethod
    def _collect_env():
        env = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
        try:
            import numpy as np
            env["numpy"] = np.__version__
        except ImportError:
            pass
        try:
            import pandas as pd
            env["pandas"] = pd.__version__
        except ImportError:
            pass
        try:
            import torch
            env["torch"] = torch.__version__
            env["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                env["cuda_version"] = torch.version.cuda
                env["gpu"] = torch.cuda.get_device_name(0)
            env["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
            env["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
            env["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        except Exception:
            pass
        try:
            env["git_commit"] = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3,
                cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
        except Exception:
            env["git_commit"] = None
        return env

    def save(self, force=False):
        if self._saving or self._t0 is None:
            return
        self._saving = True
        try:
            doc = {"status": self.status, "started_at": self._started_at,
                   "finished_at": self._finished_at,
                   "duration_s": self._duration if self._duration is not None
                   else round(time.time() - self._t0, 3)}
            for k in self._ORDER:
                if k in ("status", "started_at", "finished_at", "duration_s"):
                    continue
                if k == "error":
                    if self.error:
                        doc["error"] = self.error
                elif k in self.sections:
                    doc[k] = self.sections[k]
            for k, v in self.sections.items():
                if k not in doc:
                    doc[k] = v
            doc["console"] = list(self.console)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_jsonable(doc), f, ensure_ascii=False, indent=2, allow_nan=False)
            os.replace(tmp, self.path)
            self._last_save = time.time()
        except Exception as e:                # лог не должен ронять обучение
            if not self._warned:
                self._warned = True
                try:
                    (self._orig_err or sys.__stderr__).write(f"[RunLog] WARNING: не удалось сохранить лог: {e!r}\n")
                except Exception:
                    pass
        finally:
            self._saving = False