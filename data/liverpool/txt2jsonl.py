import ast
import json
import re
from pathlib import Path

import polars as pl

p = re.compile('(?<!\\\\)\'')


def do_one_file(in_path: Path, out_path: Path) -> None:
    """
    Convert a txt file to a jsonl file.
    """
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # lines = [json.loads(line) for line in lines]

    temp = []
    for i, line in enumerate(lines):
        try:
            line = ast.literal_eval(line)
        except:
            try:
                line = p.sub('\"', line.replace('"', r'\"')).replace(r"\'", "'")
                line = json.loads(line)
            except:
                print(f"Error decoding line {i}") # : {line}")
                continue
        temp.append(line)
    lines = temp

    df = pl.DataFrame(lines)
    df.write_ndjson(out_path)


def main(in_prefix: str, in_folder: Path) -> None:
    """
    Convert all txt files in a folder to jsonl files.
    """
    for in_file in in_folder.glob(f"{in_prefix}*.txt"):
        out_file = in_file.with_suffix(".jsonl")
        do_one_file(in_file, out_file)
        print(f"Converted {in_file} to {out_file}")


if __name__ == "__main__":
    in_folder = Path.cwd()
    in_prefix = "LiverpoolFC_"

    main(in_prefix, in_folder)
