from pathlib import Path

from setuptools import setup, find_packages


def read_requirements(path):
    req_file = Path(path)
    if not req_file.exists():
        return []
    
    # Keep only lines that are valid dependencies
    lines = req_file.read_text(encoding="utf-8").splitlines()
    requirements = []
    for line in lines:
        line = line.strip()
        # Skip comments, blank lines, and includes of other files (pip handles those itself)
        if line and not line.startswith("#") and not line.startswith("-r"):
            requirements.append(line)
    return requirements


# Load the README safely
try:
    long_description = Path("README.md").read_text(encoding="utf-8")
except FileNotFoundError:
    long_description = ""


# Keep setup.py aligned with the package runtime dependencies
install_requires = read_requirements("requirements/runtime.txt") or ["numpy"]
extras_require = {
    "test": read_requirements("requirements/test.txt"),
    "notebook": read_requirements("requirements/notebook.txt"),
    "gpu": read_requirements("requirements/gpu.txt"),
    "docs": read_requirements("requirements/docs.txt"),
}

setup(
    name="baorecon",
    version="0.5.0",
    author="Edoardo Maragliano",
    author_email="edoardo.maragliano@edu.unige.it",
    description="A simple package to perform Zeldovich reconstruction of density and velocity fields.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EdoardoMaragliano/baorecon",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
)
