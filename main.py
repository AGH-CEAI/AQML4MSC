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


def main():
    print("Hello from aqml4msc!")
    aqml_hpo()


if __name__ == "__main__":
    main()
