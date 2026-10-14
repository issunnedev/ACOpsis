import os
import subprocess
import sys
import venv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"
CACHE_DIR = PROJECT_DIR / "youtube_cache"
ENV_FILE = PROJECT_DIR / ".env"
GATHER_SCRIPT = PROJECT_DIR / "gather.py"

DEPENDENCIES = [
    "requests",
    "python-dotenv"
    "yt-dlp",
]


def run(command):
    """Execute a command and stop if it fails."""
    print("\n>", subprocess.list2cmdline(map(str, command)))
    subprocess.run(command, check=True, cwd=PROJECT_DIR)


def get_venv_python():
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def prepare_files():
    # Create/update requirements.txt with required packages.
    REQUIREMENTS.write_text(
        "\n".join(DEPENDENCIES) + "\n",
        encoding="utf-8",
    )
    print("requirements.txt ready.")

    # Preserve existing API key and other environment settings.
    if not ENV_FILE.exists():
        ENV_FILE.write_text(
            "# YouTube Data API v3 key\n"
            "YOUTUBE_API_KEY=\n",
            encoding="utf-8",
        )
        print(".env created. Add your YouTube API key.")
    else:
        print(".env already exists; preserved.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory ready: {CACHE_DIR}")

    gitignore = PROJECT_DIR / ".gitignore"
    ignored = [".venv/", ".env", "__pycache__/", "*.py[cod]"]

    existing = (
        gitignore.read_text(encoding="utf-8").splitlines()
        if gitignore.exists()
        else []
    )

    missing = [item for item in ignored if item not in existing]

    if missing:
        with gitignore.open("a", encoding="utf-8") as file:
            if existing and existing[-1].strip():
                file.write("\n")
            file.write("\n".join(missing) + "\n")


def install_dependencies(python):
    # Upgrade the old bundled pip first.
    run([
        str(python),
        "-m", "ensurepip",
        "--upgrade",
    ])

    run([
        str(python),
        "-m", "pip",
        "install",
        "--upgrade",
        "--index-url",
        "https://pypi.org/simple",
        "pip",
    ])

    # Install the latest compatible package versions from PyPI.
    run([
        str(python),
        "-m", "pip",
        "install",
        "--upgrade",
        "--index-url",
        "https://pypi.org/simple",
        "-r",
        str(REQUIREMENTS),
    ])

    # Verify imports using the same Python interpreter.
    run([
        str(python),
        "-c",
        "import requests, dotenv; "
        "print('requests:', requests.__version__); "
        "print('python-dotenv: installed successfully')",
    ])

    # Explicitly verify imports in the exact interpreter used to run gather.py.
    run([
        str(python),
        "-c",
        "import requests, dotenv; "
        "print('requests:', requests.__version__); "
        "print('python-dotenv: import OK')",
    ])


def main():
    print("=== YouTube Collector Setup ===")
    print(f"Project: {PROJECT_DIR}")

    prepare_files()

    if not VENV_DIR.exists():
        print("\nCreating virtual environment...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)

    python = get_venv_python()

    if not python.exists():
        raise RuntimeError(
            f"Virtual environment Python not found: {python}"
        )

    print("\nInstalling and verifying dependencies...")
    install_dependencies(python)

    print("\n=== Setup successful ===")
    print(f"Python: {python}")
    print(f"Collector: {GATHER_SCRIPT}")

    if not GATHER_SCRIPT.exists():
        print(
            "\nWarning: gather.py was not found. "
            "Place it in the project directory to run it here."
        )
        return

    answer = input("\nRun gather.py now? [y/N]: ").strip().lower()

    if answer in ("y", "yes"):
        # Run with the same interpreter where dependencies were installed.
        run([str(python), str(GATHER_SCRIPT)])
    else:
        print("\nTo run it later, use:")
        print(f'"{python}" "{GATHER_SCRIPT}"')


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(
            f"\nSetup command failed (exit code {error.returncode}). "
            "Check the error above and try again."
        )
    except (OSError, RuntimeError) as error:
        sys.exit(f"\nSetup failed: {error}")
    except KeyboardInterrupt:
        sys.exit("\nSetup cancelled.")