"""Sampled runtime diagnostics, kept outside checkpoint cleanup directories."""

from contextvars import ContextVar
import csv
from functools import wraps
import inspect
from pathlib import Path
import time

import torch
from torch._subclasses.fake_tensor import FakeTensor


_RUN = ContextVar("training_diagnostics_run", default=None)
_ACTIVE = ContextVar("training_diagnostics_sample", default=None)
OPERATOR_FIELDS = (
    "optimizer_step", "method", "orthogonalize", "layer", "rank",
    "b_frobenius", "q_frobenius", "b_orthogonality_error", "q_orthogonality_error",
)
TIMING_FIELDS = (
    "optimizer_step", "micro_step", "forward_id", "method", "phase",
    "batch_size", "sequence_length", "steering_call_count",
    "steering_application_ms", "model_forward_ms", "timing_backend",
)
DIAGNOSTICS_DEFINITIONS = {
    "cadence": "first optimizer update and every train_diagnostics_gap updates; 0 disables",
    "operator_state": "effective factors at initialization, after sampled updates, and final update",
    "model_forward": "underlying model through logits, including steering/runtime hooks; stops before traced loss/backward",
    "steering": "application to activations including scaling/clipping; excludes operator construction",
    "excluded": "backward, optimizer, post-update retraction, scanning, validation, PCA, final evaluation",
    "units": "milliseconds",
}


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        value = value.values()
    if isinstance(value, (list, tuple)) or type(value).__name__ == "dict_values":
        for item in value:
            found = _tensor(item)
            if found is not None:
                return found
    return None


def _real(value):
    tensor = _tensor(value)
    return tensor is not None and not isinstance(tensor, FakeTensor) and tensor.device.type != "meta"


class _Span:
    def __init__(self, device):
        self.device = torch.device(device)
        self.end = None
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self.start = torch.cuda.Event(enable_timing=True)
                self.start.record(torch.cuda.current_stream(self.device))
        else:
            self.start = time.perf_counter()

    def stop(self):
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self.end = torch.cuda.Event(enable_timing=True)
                self.end.record(torch.cuda.current_stream(self.device))
        else:
            self.end = time.perf_counter()

    def milliseconds(self, synchronize=False):
        if self.end is None:
            raise RuntimeError("Incomplete training timing span")
        if self.device.type == "cuda":
            if synchronize:
                self.end.synchronize()
            return self.start.elapsed_time(self.end)
        return (self.end - self.start) * 1000.0


def _begin_runtime(value, recorder):
    # nnsight validates nodes using fake tensors; never count those executions.
    if not _real(value) or not recorder.current_forward:
        return value, None
    span = _Span(_tensor(value).device)
    recorder.current_forward["steering"].append(span)
    return value, span


def _end_runtime(value, span):
    if span is not None and _real(value):
        span.stop()
    return value


def _forward_output_runtime(value, recorder):
    frame = recorder.current_forward
    if _real(value) and frame is not None and frame["span"].end is None:
        # nnsight can execute backward inside an lm_head output hook, before
        # the root module returns. Stop at the logits dependency so downstream
        # loss/backward nodes cannot be included in the forward measurement.
        frame["span"].stop()
    return value


def _runtime_call(fn, *args):
    if isinstance(args[0], (torch.Tensor, tuple, list)):
        return fn(*args)
    # proxy_wrapper creates a runtime node when given a nnsight Proxy. Its
    # output depends on the input activation, preserving execution ordering.
    from nnsight.tracing.Proxy import proxy_wrapper
    return proxy_wrapper(fn)(*args)


def steering_begin(value):
    recorder = _ACTIVE.get()
    if recorder is None:
        return value, None
    marked = _runtime_call(_begin_runtime, value, recorder)
    return marked[0], marked[1]


def steering_end(value, token):
    if token is None:
        return value
    return _runtime_call(_end_runtime, value, token)


def operator_metrics(basis, rotation):
    """Detached float32 diagnostics on the effective, potentially raw factors."""
    with torch.no_grad():
        basis, rotation = basis.detach().float(), rotation.detach().float()
        identity = torch.eye(basis.shape[-1], device=basis.device)
        values = (
            torch.linalg.matrix_norm(basis, ord="fro"),
            torch.linalg.matrix_norm(rotation, ord="fro"),
            torch.linalg.matrix_norm(basis.T @ basis - identity, ord="fro"),
            torch.linalg.matrix_norm(rotation.T @ rotation - identity, ord="fro"),
        )
        return dict(zip(OPERATOR_FIELDS[5:], (value.item() for value in values)))


class TrainingDiagnostics:
    def __init__(self, log_dir, method, gap=100, model=None):
        if gap < 0:
            raise ValueError("train_diagnostics_gap must be >= 0")
        self.enabled = bool(log_dir) and gap > 0
        self.log_dir = Path(log_dir) if log_dir else None
        self.method, self.gap, self.model = method, gap, model
        self.operation = None
        self.last_update = 0
        self.last_operator_step = None
        self.phase = "training"
        self.records, self.frames, self.handles = [], [], []
        self.files = {}
        self.sample_token = None
        self.forward_id = 0

    def __enter__(self):
        self.run_token = _RUN.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.end_microbatch()
            if exc_type is None:
                self.log_operators(self.last_update, final=True)
        finally:
            for file, _ in self.files.values():
                file.close()
            _RUN.reset(self.run_token)

    def _write(self, filename, fields, row):
        if filename not in self.files:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            # Never replace existing training logs, including evaluation reuse.
            file = (self.log_dir / filename).open("x", newline="", encoding="utf-8")
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            self.files[filename] = file, writer
        file, writer = self.files[filename]
        writer.writerow(row)
        file.flush()

    def sampled(self, step):
        return self.enabled and (step == 1 or step % self.gap == 0)

    def attach_operation(self, operation):
        self.operation = operation
        self.log_operators(0, final=True)

    def log_operators(self, step, final=False):
        self.last_update = step
        if (not self.enabled or self.method not in {"activation_additive_rot", "shtiefel_additive_rot"}
                or self.operation is None or self.last_operator_step == step
                or not (final or self.sampled(step))):
            return
        op = self.operation
        with torch.no_grad():
            for layer in sorted(op._optimized_layer_idxs):
                basis, rotation = op._subspace_factors(layer)
                self._write("operator_metrics.csv", OPERATOR_FIELDS, {
                    "optimizer_step": step, "method": self.method,
                    "orthogonalize": op.orthogonalize_enabled, "layer": layer,
                    "rank": basis.shape[-1], **operator_metrics(basis, rotation),
                })
        self.last_operator_step = step

    @property
    def current_forward(self):
        return self.frames[-1] if self.frames else None

    def forward_output(self, value):
        if self.sample_token is None:
            return value
        return _runtime_call(_forward_output_runtime, value, self)

    def _pre_forward(self, module, args, kwargs):
        inputs = (args, kwargs)
        if not _real(inputs):
            self.frames.append(None)
            return
        tensor = _tensor(inputs)
        # Token inputs can still be on CPU when a model dispatches them to its
        # accelerator. Time the model's device, not the tokenizer's device.
        parameter = next(module.parameters(), None)
        device = parameter.device if parameter is not None else tensor.device
        self.forward_id += 1
        self.frames.append({
            "span": _Span(device), "steering": [],
            "optimizer_step": self.optimizer_step, "micro_step": self.micro_step,
            "forward_id": self.forward_id, "method": self.method, "phase": self.phase,
            "batch_size": tensor.shape[0] if tensor.ndim else 1,
            "sequence_length": tensor.shape[1] if tensor.ndim > 1 else 1,
        })

    def _post_forward(self, module, args, kwargs, output):
        frame = self.frames.pop()
        if frame is None:
            return
        if frame["span"].end is None:
            frame["span"].stop()
        # A failed forward must not be represented as a valid timing sample.
        if output is None:
            return
        self.records.append(frame)

    def begin_microbatch(self, optimizer_step, micro_step):
        self.end_microbatch()
        if not self.sampled(optimizer_step):
            return
        self.optimizer_step, self.micro_step = optimizer_step, micro_step
        self.sample_token = _ACTIVE.set(self)
        try:
            self.handles.append(self.model.register_forward_pre_hook(self._pre_forward, with_kwargs=True, prepend=True))
            self.handles.append(self.model.register_forward_hook(self._post_forward, with_kwargs=True, always_call=True))
        except BaseException:
            self.end_microbatch()
            raise

    def end_microbatch(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.frames.clear()
        if self.sample_token is not None:
            _ACTIVE.reset(self.sample_token)
            self.sample_token = None
        records, self.records = self.records, []
        for frame in records:
            span, steering = frame.pop("span"), frame.pop("steering")
            model_ms = span.milliseconds(synchronize=True)
            self._write("training_forward_timing.csv", TIMING_FIELDS, {
                **frame, "steering_call_count": len(steering),
                "steering_application_ms": sum(s.milliseconds() for s in steering),
                "model_forward_ms": model_ms,
                "timing_backend": "cuda_event" if span.device.type == "cuda" else "cpu_perf_counter",
            })


def current_diagnostics():
    return _RUN.get()


def with_training_diagnostics(fn):
    """Ensure hooks and log files close even when training raises an exception."""
    signature = inspect.signature(fn)

    @wraps(fn)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        checkpoint_dir = bound.arguments.get("tb_checkpoint_dir")
        log_dir = Path(checkpoint_dir).parent if checkpoint_dir else None
        gap = bound.arguments.get("train_diagnostics_gap", 100)
        method = bound.arguments["direction_mode"]
        model = bound.arguments["model"]
        actual = getattr(model, "_model", model)
        with TrainingDiagnostics(log_dir, method, gap, actual):
            return fn(*args, **kwargs)

    return wrapped
