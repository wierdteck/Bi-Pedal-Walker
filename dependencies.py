import subprocess
import sys

def pip_install(packages):
    """Install a list of packages via pip."""
    for pkg in packages:
        print(f"\n📦 Installing: {pkg}")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            pkg
        ])

if __name__ == "__main__":
    print("🚀 Installing dependencies for PPO + BipedalWalkerHardcore...\n")

    packages = [
        # Core scientific stack
        "numpy",
        "matplotlib",

        # PyTorch (CPU version by default)
        "torch",

        # Gymnasium + Box2D environments (REQUIRED for BipedalWalker)
        "gymnasium[box2d]",
        "box2d-py",

        "stable-baselines3[extra]"
    ]

    pip_install(packages)

    print("\n✅ All dependencies installed successfully!")
    print("👉 progress_bar=True will now work correctly.")
