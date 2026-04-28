from pathlib import Path

def mkdir(path):
    path.mkdir(parents=True, exists_ok=True)
