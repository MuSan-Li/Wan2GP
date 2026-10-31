"""Build and validate candidate environments independently of active application code."""

import json
import os
from pathlib import Path

from .environments import interpreter_path
from .execute import ProcessRunner
from .state import InstallError, digest, file_hash, plain_path


INVENTORY_SCRIPT = """
import importlib.metadata, json, re, sys
packages = {re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version for d in importlib.metadata.distributions()}
print(json.dumps({"python": sys.version.split()[0], "prefix": sys.prefix, "base_prefix": sys.base_prefix, "packages": packages}))
"""

COMPUTE_SCRIPT = """
import json, sys, torch
backend = sys.argv[1]
if backend == "mps":
    assert torch.backends.mps.is_available(), "MPS is unavailable"
    device = "mps"
else:
    assert torch.cuda.is_available(), "Selected GPU runtime is unavailable"
    assert (torch.version.hip is not None) == (backend == "rocm"), "Wrong Torch accelerator build"
    device = "cuda:0"
x = torch.ones((2, 2), device=device)
result = (x @ x).float().cpu()
assert torch.isfinite(result).all() and torch.equal(result, torch.full((2, 2), 2.0))
print(json.dumps({"backend": backend, "torch": torch.__version__, "compute": "passed"}))
"""


class EnvironmentService:
    def __init__(self, runtime, runner=None, cache_directory=None, kind="uv"):
        self.runtime = runtime
        self.runner = runner or ProcessRunner()
        self.cache_directory = plain_path(cache_directory) if cache_directory is not None else None
        self.kind = kind

    def inventory(self, environment):
        environment = plain_path(environment)
        python = interpreter_path(environment, self.kind)
        result = self.runner.run([python, "-I", "-B", "-c", INVENTORY_SCRIPT], env=self.runtime.environment(python), timeout=30, capture=True, emit_output=False)
        inventory = json.loads(result.stdout)
        if not os.path.samefile(plain_path(inventory["prefix"]), environment) or (self.kind != "conda" and os.path.samefile(inventory["prefix"], inventory["base_prefix"])):
            raise InstallError("wrong_environment", f"Interpreter does not belong to its managed environment: {environment}")
        if self.kind == "conda" and not (environment / "conda-meta").is_dir():
            raise InstallError("wrong_environment", "The selected interpreter is not a Conda environment")
        return inventory

    def build(self, environment, resolution_directory, resolution):
        environment = plain_path(environment)
        environment.mkdir(parents=True, exist_ok=False)
        command = self.runtime.create_command(environment, resolution["python"], seed=False)
        self.runner.run(command, env=self.runtime.environment())
        python = interpreter_path(environment, self.kind)
        lock = plain_path(resolution_directory) / "pylock.toml"
        if file_hash(lock) != resolution["files"]["pylock.toml"]:
            raise InstallError("changed_input", "Dependency lock changed before installation")
        command = self.runtime.command("pip", "sync", "--python", python, "--require-hashes", str(lock))
        if resolution.get("no_index", False):
            command.append("--no-index")
        package_env = self.runtime.environment(python)
        if self.cache_directory is not None:
            # Keep source builds short on Windows and package links on the target drive.
            package_env["UV_CACHE_DIR"] = str(self.cache_directory)
        self.runner.run(command, env=package_env)
        return self.validate(environment, resolution)

    def validate(self, environment, resolution, expected_inventory=None):
        inventory = self.inventory(environment)
        if inventory["python"] != resolution["python"]:
            raise InstallError("python_mismatch", "Candidate interpreter version differs from its resolution")
        if expected_inventory is not None and inventory != expected_inventory:
            raise InstallError("environment_changed", "Environment packages or interpreter changed; reconcile manual/plugin changes before updating")
        python = interpreter_path(environment, self.kind)
        self.runner.run(self.runtime.command("pip", "check", "--python", python), env=self.runtime.environment(python), timeout=60)
        health = {"packages": "checked", "compute": "not_applicable"}
        if "torch" in inventory["packages"]:
            profile = resolution["profile"]
            backend = "mps" if profile == "MPS" else "rocm" if profile == "AMD" else "cuda"
            env = self.runtime.environment(python)
            if backend != "mps":
                device = resolution["selected_device"]
                if not device:
                    raise InstallError("device_required", "Select a device when preparing this GPU installation")
                if backend == "cuda" and not device["id"].startswith("GPU-"):
                    raise InstallError("device_required", "CUDA validation requires an NVIDIA GPU UUID")
                if backend == "rocm" and "index" not in device:
                    raise InstallError("device_required", "ROCm validation requires an explicit runtime device index")
                env["CUDA_VISIBLE_DEVICES" if backend == "cuda" else "HIP_VISIBLE_DEVICES"] = device["id"] if backend == "cuda" else str(device["index"])
            result = self.runner.run([python, "-I", "-c", COMPUTE_SCRIPT, backend], env=env, timeout=90, capture=True)
            health["compute"] = json.loads(result.stdout)
        return {"inventory": inventory, "inventory_sha256": digest(inventory), "health": health}
