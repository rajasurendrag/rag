import shutil
import subprocess
import time

import ollama

from app.config import CHROMA_PERSISTENCE_DIR, EMBEDDING_MODEL, LLM_MODEL


OLLAMA_STARTUP_TIMEOUT_SECONDS = 30
OLLAMA_POLL_INTERVAL_SECONDS = 0.5


def is_ollama_running(client):
  try:
    client.list()
    return True
  except Exception:
    return False


def start_ollama():
  ollama_path = shutil.which("ollama")

  if ollama_path is None:
    raise RuntimeError(
      "Ollama isn't installed. Install it with: brew install ollama"
    )

  log_path = CHROMA_PERSISTENCE_DIR.parent / "ollama.log"
  log_path.parent.mkdir(parents=True, exist_ok=True)

  with open(log_path, "a") as log_file:
    subprocess.Popen(
      [ollama_path, "serve"],
      stdout=log_file,
      stderr=log_file,
      start_new_session=True,
    )


def ensure_ollama_running():
  client = ollama.Client()

  if is_ollama_running(client):
    return

  print("Ollama isn't running, starting it now...")
  start_ollama()

  deadline = time.monotonic() + OLLAMA_STARTUP_TIMEOUT_SECONDS

  while time.monotonic() < deadline:
    if is_ollama_running(client):
      return

    time.sleep(OLLAMA_POLL_INTERVAL_SECONDS)

  raise RuntimeError(
    "Timed out waiting for Ollama to start. Try running 'ollama serve' manually."
  )


def ensure_model_pulled(model):
  client = ollama.Client()

  installed = {entry.model.split(":")[0] for entry in client.list().models}

  if model in installed:
    return

  print(f"Pulling model '{model}', this may take a few minutes...")

  last_percent = None

  for progress in client.pull(model, stream=True):
    if not (progress.total and progress.completed):
      continue

    percent = int(progress.completed / progress.total * 100)

    if percent != last_percent:
      print(f"\r{progress.status}: {percent}%", end="", flush=True)
      last_percent = percent

  print()


def ensure_ollama_ready():
  ensure_ollama_running()

  for model in (EMBEDDING_MODEL, LLM_MODEL):
    ensure_model_pulled(model)
