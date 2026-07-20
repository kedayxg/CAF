import sys

from experiments.classical.run_scaling import main as scaling_main


def main():
    extra = []
    argv = sys.argv[1:]
    if "--sizes" not in argv:
        extra += ["--sizes", "200"]
    if "--k-ratios" not in argv:
        extra += ["--k-ratios", "0.25,0.5"]
    if "--solvers" not in argv:
        extra += ["--solvers", "sa,gurobi"]
    if "--out-dir" not in argv:
        extra += ["--out-dir", "outputs_sparsity_sweep"]
    sys.argv = [sys.argv[0], *argv, *extra]
    scaling_main()


if __name__ == "__main__":
    main()
