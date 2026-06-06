# src/config/_config.py
from __future__ import annotations
import dataclasses as dcls
from enum import Enum, auto
import functools as ft
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from typing_extensions import override
# TODO fix - src.core imported jax, causing timeout
# from src.core import ReplaceMixin
from src.core.static_types import ReplaceMixin
from src.config._info import LIBRARY_NAME, ENV_PREFIX, DEFAULT_USER, DEFAULT_GROUP


class ModuleDirs(Enum):
  @staticmethod
  @override
  def _generate_next_value_(name: str, start: int, count: int, last_values: list[Any]): 
    return f"{name.upper()}_DIR"
  
  PROJECT       = auto()
  SRC           = auto()
  DATA          = auto()
  LOG_ROOT      = auto()
  RUN_ROOT      = auto()
  TBOARD        = auto()
  IMAGES        = auto()
  SHARED_LOGS   = auto()
  EXPERIMENTS   = auto()
  MEDIA         = auto()
  TESTS         = auto()
  TEST_FIXTURES = auto()
  CONDOR        = auto()
  


class RemoteDirs(Enum):
  @staticmethod
  @override
  def _generate_next_value_(name: str, start: int, count: int, last_values: list[Any]):
    return f"{name.upper()}_REMOTE"

  GW_HOST = auto()
  GW_BASE = auto()
  

@dcls.dataclass(frozen=True)
class GWRemote:
  host: str
  base_dir: Path

@dcls.dataclass(frozen=True)
class Defaults(ReplaceMixin):
  cpus: int = 4
  mem: str = "96GB"
  disk: str = "4GB"

  # thread defaults
  blas_threads: int = 2      # per-process BLAS/OpenMP cap
  xla_threads: int = 4       # XLA intra-op threads cap
  # TODO needed? need more?
  io_threads: int = 2

@dcls.dataclass(frozen=True)
class Settings(ReplaceMixin):
  module_name: str = LIBRARY_NAME
  # identity / accounting
  user: str = DEFAULT_USER
  group: str = DEFAULT_GROUP
  device: str = "cpu"  # "cpu" | "gpu" | "auto" (TODO)
 
  # host-side project layout
  project_dir: Path = Path(".")         # host path to repo root (bound to /cem in container)
  src_root: Path = Path("src") # src
  run_root: Path = Path("var/runs")     # default: project_dir / "var" / "runs"
  log_root: Path = Path("var/logs")      # default: project_dir / "var" / "logs"
  exp_dir: Path = Path("experiments")  # host path for logs, etc.
  image: Path = Path("images/cem_sbx_cpu")           # host path to apptainer images
  condor_dir: Path = Path("condor")           # project_dir / "condor"
  shared_logs_dir: Path = Path("var/condor")      # host path for shared event.log
  tboard_dir: Path = Path("var/tboard")  # tensorboard directory
  data_dir: Path = Path("data")
  media_dir: Path = Path("media")
  tests_dir: Path = Path("tests")
  test_fixtures_dir: Path = Path("tests/fixtures")

  # container-side fixed mount
  container_root: Path = Path(f"/{LIBRARY_NAME}")

  # submit defaults
  defaults: Defaults = dcls.field(default_factory=Defaults)

  remote: GWRemote | None = None

  @property
  def event_log(self) -> Path:
    # Single shared file; for per-campaign, create at runtime instead
    return self.shared_logs_dir / "event.log"

  @ft.cached_property
  def to_container(self, host_path: Path) -> Path:
    htc = self.host_to_container
    return htc(host_path)

  @property
  def host_to_container(self):
    r""" curried conversion method """
    proj = self.project_dir.resolve()
    root = self.container_root
    def f(path: Path) -> Path:
      return root / path.resolve().relative_to(proj)
    return f
  
  def experiment_logs(self, name: str) -> Path:
    return self.exp_dir / name / "logs"

  def perm_logs(self, name: str) -> Path:
    return self.exp_dir / name / "logs" / "perm"




def _git_root(start: Path) -> Path | None:
  try:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(start), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True
    ).stdout.strip()
    return Path(out) if out else None
  except Exception:
    return None
  

def _read_json(path: Path) -> dict[str, Any]:
  try:
    return json.loads(path.read_text())
  except Exception:
    return {}
    
def first_existing(*candidates: Path) -> Path | None:
  for p in candidates:
    if p and p.exists():
      return p
  return None

def _env(key: str|int, cfg: dict[str, Any], default: str|int="") -> str:
  # resolve env like CEM_USER, CEM_PROJECT_DIR, etc.
  ek = f"{ENV_PREFIX}{key}"
  return os.environ.get(ek) or cfg.get(ek) or str(default)



@ft.lru_cache(maxsize=1)
def _get_settings_cached() -> Settings:
  """
  Resolve settings with precedence:
    1) ENV: CEM_USER, CEM_GROUP, CEM_DEVICE, CEM_PROJECT_DIR, CEM_IMAGE, ...
    2) Config file: $CEM_CONFIG (JSON) with keys matching ENV names (CEM_*)
    3) Auto-detect project root via git, else parents of this file
    4) Internal defaults
  """
  # optional config file
  cfg = {}
  cfg_path = os.environ.get(f"{ENV_PREFIX}CONFIG")
  cfg = _read_json(Path(cfg_path)) if cfg_path else {}

  # identity/accounting
  user   = _env("USER",   cfg, os.environ.get("USER", DEFAULT_USER))
  group  = _env("GROUP",  cfg, os.environ.get("GROUP", DEFAULT_GROUP))
  device = _env("DEVICE", cfg, "cpu")

  # autodetect project root
  here = Path(__file__).resolve()
  auto_root = _git_root(here) or next((p for p in here.parents if (p / ".git").exists()), None) or here.parents[2]
  image_name = "cem_sbx_gpu" if device == "gpu" else "cem_sbx_cpu"

  project_dir = Path(_env(ModuleDirs.PROJECT.value, cfg, default=str(auto_root))).resolve()

  # derived paths (allow ENV override for image/log paths)
  def init_pth(key: str|int|ModuleDirs, default: str) -> Path:
    if isinstance(key, ModuleDirs): 
      key = key.value
    raw = _env(key, cfg, default)
    p = Path(raw)
    if not p.is_absolute():
      p = project_dir / p
    return p.resolve()
  
  MD = ModuleDirs
  src_root     = init_pth(MD.SRC,           f"src")
  run_root     = init_pth(MD.RUN_ROOT,      f"var/runs")
  log_root     = init_pth(MD.LOG_ROOT,      f"var/logs")
  condor_dir   = init_pth(MD.CONDOR,        f"condor")
  image        = init_pth(MD.IMAGES,        f"images/{image_name}")
  shared_logs  = init_pth(MD.SHARED_LOGS,   f"var/condor")
  tboard_dir   = init_pth(MD.TBOARD,        f"var/tboard")
  media_dir    = init_pth(MD.MEDIA,         f"var/media")
  exp_dir      = init_pth(MD.EXPERIMENTS,   f"experiments")
  data_dir     = init_pth(MD.DATA,          f"data")
  tests_dir    = init_pth(MD.TESTS,         f"tests")
  fixtures_dir = init_pth(MD.TEST_FIXTURES, f"tests/fixtures")

  auto_create = _env("AUTO_CREATE_DIRS", cfg, "1") == "1"
  if auto_create:
     # create “safe” dirs (never touch images)
    for p in (shared_logs, exp_dir, run_root, log_root, tboard_dir, data_dir, media_dir, tests_dir, fixtures_dir):
      p.mkdir(parents=True, exist_ok=True)
  
  gw_remote = GWRemote(
    host = _env(RemoteDirs.GW_HOST.value, cfg, "ldas-pcdev2.ligo.caltech.edu"),
    base_dir = Path(_env(RemoteDirs.GW_BASE.value, cfg, f"/home/{user}/public_html/O3/LIV")),
  )

  num_cpus = _env("CPUS", cfg, "4")
  defaults = Defaults(
    cpus         = int(num_cpus),
    mem          = _env("MEM", cfg, "96GB"),
    disk         = _env("DISK", cfg, "4GB"),
    blas_threads = int(_env("BLAS_THREADS", cfg, default=int(2))),
    xla_threads  = int(_env("NUM_THREADS",  cfg, default=num_cpus)),
    io_threads   = int(_env("IO_THREADS",   cfg, default=int(2))),
  )

  return Settings(
    user=user,
    group=group,
    device=device,
    project_dir=project_dir,
    src_root=src_root,
    run_root=run_root,
    log_root=log_root,
    condor_dir=condor_dir,
    image=image,
    exp_dir=exp_dir,
    shared_logs_dir=shared_logs,
    tboard_dir=tboard_dir,
    data_dir=data_dir,
    media_dir=media_dir,
    tests_dir=tests_dir,
    test_fixtures_dir=fixtures_dir,
    remote=gw_remote,
    defaults=defaults,
  )

def get_settings(*, refresh: bool = False) -> Settings:
  """
  Resolve settings with precedence:
    1) ENV: CEM_USER, CEM_GROUP, CEM_DEVICE, CEM_PROJECT_DIR, CEM_IMAGE, ...
    2) Config file: $CEM_CONFIG (JSON) with keys matching ENV names (CEM_*)
    3) Auto-detect project root via git, else parents of this file
    4) Internal defaults
  """
  if refresh:
    _get_settings_cached.cache_clear()
  return _get_settings_cached()


def ensure_layout(settings: Settings):
  cfg = {}
  cfg_path = os.environ.get(f"{ENV_PREFIX}CONFIG")
  cfg = _read_json(Path(cfg_path)) if cfg_path else {}
  auto_create = _env("AUTO_CREATE_DIRS", cfg, "1") == "1"
  if auto_create:
     # create “safe” dirs (never touch images)
    for p in (settings.shared_logs_dir, settings.exp_dir, settings.run_root, 
              settings.log_root, settings.tboard_dir, settings.data_dir, settings.media_dir, 
              settings.tests_dir, settings.test_fixtures_dir):
      p.mkdir(parents=True, exist_ok=True)
  return






AUTO_BUILT_DIRECTORIES: list[ModuleDirs] = [
  ModuleDirs.DATA,
  ModuleDirs.LOG_ROOT,
  ModuleDirs.RUN_ROOT,
  ModuleDirs.TBOARD,
  ModuleDirs.MEDIA,
  ModuleDirs.TESTS,
  ModuleDirs.TEST_FIXTURES,
]

