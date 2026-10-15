from pathlib import Path


# zeta_compatibility_check.py
# Locate and read ACOpsis/zeta/version from ε-Epsilon/EZL/


def find_zeta_version():
    # This file is located in ε-Epsilon/EZL/
    script_dir = Path(__file__).resolve().parent

    # EZL -> ε-Epsilon -> ACOpsis
    project_root = script_dir.parent.parent

    version_file = project_root / "zeta" / "version"

    if not version_file.is_file():
        raise FileNotFoundError(
            f"Zeta version file not found: {version_file}"
        )

    return version_file


def get_zeta_version():
    version_file = find_zeta_version()
    return version_file.read_text(encoding="utf-8").strip()


def main():
    try:
        version_file = find_zeta_version()
        version = get_zeta_version()

        print(f"Zeta version file: {version_file}")
        print(f"Zeta version: {version}")

        return 0

    except (OSError, UnicodeError) as exc:
        print(f"Zeta compatibility check failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
