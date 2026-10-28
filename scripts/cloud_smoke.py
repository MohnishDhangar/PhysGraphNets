"""One-command smoke test for Kaggle or Colab; also usable on local CPU."""
from physgraph.experiments.smoke import run_cloud_smoke


def main() -> None:
    dest = run_cloud_smoke()
    print("PASS: Phase 1 + 2A cloud/local CPU numerical smoke test")
    print(f"Results: {dest.resolve()}")


if __name__ == "__main__":
    main()
