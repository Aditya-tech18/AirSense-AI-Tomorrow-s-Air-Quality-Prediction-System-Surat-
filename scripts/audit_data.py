"""Audit the original MP252 data and optionally refresh the findings data section."""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from airsense.config import load_data_config, read_yaml_mapping
from airsense.data.quality_checks import audit_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--write", action="store_true", help="Update data_issues in findings.yaml")
    args = parser.parse_args()
    paths, rules = load_data_config(args.params)
    frames = [
        pd.read_csv(paths[split], dtype={"record_id": "string"})
        for split in ("train", "later", "holdout")
    ]
    issues = audit_training_data(*frames, rules)
    print(yaml.safe_dump(issues, sort_keys=False), end="")
    if args.write:
        path = paths["findings"]
        report = read_yaml_mapping(path)
        report["data_issues"] = issues
        temporary = path.with_suffix(".yaml.tmp")
        temporary.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
        temporary.replace(path)


if __name__ == "__main__":
    main()
