import subprocess
import sys
from pathlib import Path

# Manager.py is inside ε-Epsilon/
BASE_DIR = Path(__file__).resolve().parent

# All service scripts and their .env live here.
SERVICE_DIR = BASE_DIR / "Fetch_service"

# Cache lives inside Fetch_service.
CACHE_DIR = SERVICE_DIR / "youtube_cache"

SCRIPTS = {
    "1": {
        "name": "Setup",
        "file": "setup.py",
        "description": "Configure the service",
    },
    "2": {
        "name": "Gather",
        "file": "gather.py",
        "description": "Gather data and YouTube links",
    },
    "3": {
        "name": "Parse",
        "file": "parser.py",
        "description": "Parse data and process videos",
    },
}


def clear_screen():
    if sys.platform == "win32":
        subprocess.run(
            "cls",
            shell=True,
            check=False,
        )
    else:
        print("\033[2J\033[H", end="")


def pause():
    try:
        input("\nPress Enter to return to the menu...")
    except (KeyboardInterrupt, EOFError):
        pass


def show_menu():
    print("=" * 60)
    print("                 ACOPSIS ε-Epsilon MANAGER")
    print("=" * 60)
    print(f"Project: {BASE_DIR}")
    print(f"Service: {SERVICE_DIR}")
    print("-" * 60)

    for key, script in SCRIPTS.items():
        print(
            f"  {key}. {script['name']:<10} "
            f"- {script['description']}"
        )

    print("  4. List YouTube cache prompts")
    print("  5. Check project files")
    print("  0. Exit")
    print("=" * 60)


def run_script(script):
    script_path = SERVICE_DIR / script["file"]

    if not script_path.is_file():
        print(f"\nERROR: Script not found:\n{script_path}")
        return

    print(f"\nLaunching {script['name']}...")
    print(f"File: {script_path}")
    print(f"Working directory: {SERVICE_DIR}")
    print("-" * 60)

    try:
        result = subprocess.run(
            [sys.executable, "-u", str(script_path)],
            cwd=SERVICE_DIR,
            check=False,
        )

        print("-" * 60)

        if result.returncode == 0:
            print(f"{script['name']} finished successfully.")
        else:
            print(
                f"{script['name']} exited with "
                f"code {result.returncode}."
            )

    except KeyboardInterrupt:
        print(f"\n{script['name']} interrupted.")

    except OSError as error:
        print(f"Could not launch script: {error}")


def list_cache_prompts():
    if not CACHE_DIR.is_dir():
        print(f"\nCache directory not found:\n{CACHE_DIR}")
        return

    folders = sorted(
        (
            path for path in CACHE_DIR.iterdir()
            if path.is_dir()
        ),
        key=lambda path: path.name,
    )

    if not folders:
        print("\nNo cache prompt folders found.")
        return

    print("\nYouTube cache prompts:")
    print("-" * 60)

    for index, folder in enumerate(folders, start=1):
        prompt = folder.name.split("_", 1)[0]
        json_count = sum(
            1 for path in folder.rglob("*.json")
            if path.is_file()
        )

        print(
            f"  {index}. Prompt {prompt} | "
            f"{folder.name} | JSON files: {json_count}"
        )


def check_project_files():
    print("\nProject file check")
    print("-" * 60)

    print(
        f"[{'OK' if SERVICE_DIR.is_dir() else 'MISSING'}] "
        f"Fetch_service/"
    )

    for script in SCRIPTS.values():
        path = SERVICE_DIR / script["file"]
        status = "OK" if path.is_file() else "MISSING"
        print(f"[{status}] {script['file']}")

    env_file = SERVICE_DIR / ".env"
    print(
        f"[{'OK' if env_file.is_file() else 'MISSING'}] "
        ".env"
    )

    print(
        f"[{'OK' if CACHE_DIR.is_dir() else 'MISSING'}] "
        "youtube_cache/"
    )


def main():
    while True:
        clear_screen()
        show_menu()

        try:
            choice = input("\nSelect an option: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting ACOpis Manager.")
            break

        if choice == "0":
            print("Goodbye!")
            break

        elif choice in SCRIPTS:
            clear_screen()
            run_script(SCRIPTS[choice])
            pause()

        elif choice == "4":
            list_cache_prompts()
            pause()

        elif choice == "5":
            check_project_files()
            pause()

        else:
            print("\nInvalid selection.")
            pause()


if __name__ == "__main__":
    main()
