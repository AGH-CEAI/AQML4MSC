from aqml4msc.experiments.ansatz_search import main as aqml_hpo

"""
from aqml4msc.experiments.baseline_hpo import (  # noqa: F401
    hpo_baseline_1,
    hpo_baseline_2,
    hpo_baseline_3,
)
from aqml4msc.experiments.manual_exp import manual_exp_1, manual_exp_2  # noqa: F401
from aqml4msc.experiments.quantum_hpo import hpo_quantum_1  # noqa: F401
"""
import os
from pathlib import Path


def main():
    print("Hello from aqml4msc!\n")
    print("Checking if all the required environmental variables are set:")
    with open("aqml4msc.env", "r") as f:
        for line in f.readlines():
            var: str = line.split(" ")[0]
            print(f"\t{var} = {os.environ[var]}")
    print("Done. Checking if data files exists:")
    for data_file_var in ["TRAIN_VAL_IMAGES_PATH", "TRAIN_VAL_LABELS_PATH"]:
        data_file_path: Path = Path(os.environ[data_file_var])
        print(f"\t{data_file_var} exists: {data_file_path.exists()}")
        if not data_file_path.exists():
            raise FileNotFoundError(f"Data file {os.environ[data_file_var]} doesn't exist! Experiment impossible!")

    print("All set. Starting experiment.")
    aqml_hpo()
    print("Experiment done.")


if __name__ == "__main__":
    main()
