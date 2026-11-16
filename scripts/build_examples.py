"""Execute the jupytext-style example scripts and render them as Markdown pages.

Zensical has no counterpart to ``mkdocs-jupyter``, so the scripts in ``examples/*.py`` are run here
and written to ``docs/examples/<name>.md`` (code, text output, plots as PNG files in
``docs/examples/<name>_files/``). The generated directory is not tracked by git. Run this before
``zensical build``::

    uv run --group docs python scripts/build_examples.py
"""

from __future__ import annotations

import base64
import shutil
import sys
from pathlib import Path

import jupytext
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "examples"
OUT = ROOT / "docs" / "examples"


def text_of(value: str | list[str]) -> str:
    return "".join(value) if isinstance(value, list) else value


def render(nb, files_dir: Path) -> str:
    out: list[str] = []
    n_img = 0
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            out.append(text_of(cell.source).strip())
            continue
        if cell.cell_type != "code":
            continue
        out.append("```python\n" + text_of(cell.source).strip() + "\n```")
        for o in cell.get("outputs", []):
            kind = o.output_type
            if kind == "stream":
                body = text_of(o.text).rstrip()
                if o.name == "stderr" and not body:
                    continue
                if body:
                    out.append('```text title="Output"\n' + body + "\n```")
            elif kind == "error":
                raise RuntimeError("\n".join(o.traceback))
            elif kind in ("display_data", "execute_result"):
                data = o.data
                if "image/png" in data:
                    n_img += 1
                    files_dir.mkdir(parents=True, exist_ok=True)
                    name = f"output_{n_img}.png"
                    (files_dir / name).write_bytes(base64.b64decode(text_of(data["image/png"])))
                    out.append(f"![Plot {n_img}]({files_dir.name}/{name})")
                elif "text/plain" in data and kind == "execute_result":
                    out.append(
                        '```text title="Output"\n' + text_of(data["text/plain"]).rstrip() + "\n```"
                    )
    return "\n\n".join(out) + "\n"


def main() -> None:
    scripts = sorted(SOURCES.glob("*.py"))
    if not scripts:
        sys.exit(f"no example scripts found in {SOURCES}")
    OUT.mkdir(parents=True, exist_ok=True)
    for path in scripts:
        files_dir = OUT / f"{path.stem}_files"
        shutil.rmtree(files_dir, ignore_errors=True)
        nb = jupytext.read(path)
        nb.metadata.setdefault(
            "kernelspec", {"name": "python3", "language": "python", "display_name": "Python 3"}
        )
        print(f"executing {path.name}", flush=True)
        NotebookClient(
            nb, kernel_name="python3", resources={"metadata": {"path": str(SOURCES)}}
        ).execute()
        (OUT / f"{path.stem}.md").write_text(render(nb, files_dir))


if __name__ == "__main__":
    main()
