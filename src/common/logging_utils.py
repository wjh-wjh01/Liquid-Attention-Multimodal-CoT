import pathlib
from datetime import datetime


def log_line(message: str, log_file: str | pathlib.Path | None = None) -> None:
    stamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(stamped)
    if log_file is not None:
        p = pathlib.Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(stamped + "\n")
