"""Choose CuRobo compilation targets from local GPUs without host packages."""
import subprocess
import sys


def main():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True)
        capabilities = sorted({float(line.strip()) for line in output.splitlines() if line.strip()})
        if not capabilities:
            raise ValueError("No GPU architectures found")
    except (OSError, ValueError, subprocess.CalledProcessError):
        print("No local GPU detected during build; using this project's RTX 2080 Ti target 7.5. Set ROBOTWIN_CUDA_ARCH explicitly for another target.", file=sys.stderr)
        capabilities = [7.5]
    if any(value > 9.0 or value < 5.0 for value in capabilities):
        raise SystemExit(f"CUDA 12.1 cannot compile detected GPU architectures {capabilities}. A newer validated CUDA/Torch stack is required; or set ROBOTWIN_CUDA_ARCH for only the supported GPUs you intend to use.")
    print(";".join(f"{value:.1f}" for value in capabilities))


if __name__ == "__main__":
    main()
