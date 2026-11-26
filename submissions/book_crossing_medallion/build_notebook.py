"""Build a standalone notebook from percent-format source; optionally execute it."""
import argparse
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def build(execute=False):
    folder = Path(__file__).resolve().parent
    source = (folder / "book_recommender.py").read_text()
    chunks = re.split(r"(?m)^# %%([^\n]*)\n", source)
    cells = []
    for index in range(1, len(chunks), 2):
        marker, text = chunks[index], chunks[index + 1].strip("\n")
        if "[markdown]" in marker:
            text = "\n".join(re.sub(r"^# ?", "", line) for line in text.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(text))
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3"
            },
            "language_info": {"name": "python"},
            "colab": {"name": "book_recommender.ipynb", "provenance": []},
        },
    )
    destination = folder / "book_recommender.ipynb"
    if execute:
        NotebookClient(
            notebook, timeout=900, kernel_name="python3",
            resources={"metadata": {"path": str(folder)}},
        ).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, destination)
    print(f"{'Executed and saved' if execute else 'Built'} {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    build(parser.parse_args().execute)
