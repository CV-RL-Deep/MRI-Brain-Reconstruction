import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless Runner for MRI Brain Reconstruction Experiment Notebook"
    )
    parser.add_argument(
        "--notebook",
        type=str,
        default="experiments/autoregressive.ipynb",
        help="Path to the target experiment notebook to execute.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="ablation",
        choices=[
            "all",
            "ablation",
            "sota",
            "dsc",
            "frechet",
            "cvpr",
            "train",
            "hpo",
            "eda",
            "viz",
        ],
        help="Execution mode injected into the experiment environment.",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Path to pre-trained weights directory.",
    )
    parser.add_argument(
        "--preload-dir",
        type=str,
        default=None,
        help="Path to preloaded results directory.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=-1,
        help="Cell execution timeout in seconds (-1 for infinite).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.notebook):
        print(
            f"Error: Target notebook '{args.notebook}' not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Propagate CLI arguments as environment variables to the executed notebook kernel
    env = os.environ.copy()
    env["EXPERIMENT_MODE"] = args.mode
    if args.weights_dir:
        env["WEIGHTS_DIR"] = args.weights_dir
    if args.preload_dir:
        env["PRELOAD_DIR"] = args.preload_dir

    output_nb = args.notebook.replace(".ipynb", "_nbexec.ipynb")

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={args.timeout}",
        f"--output={os.path.basename(output_nb)}",
        f"--output-dir={os.path.dirname(output_nb) or '.'}",
        args.notebook,
    ]

    print(f"Executing '{args.notebook}' in mode '{args.mode}' via nbconvert...")
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
