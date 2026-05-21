from pathlib import Path

from setuptools import setup, find_packages


def read_requirements(path):
    try:
        return [
            line.strip()
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("-r")
        ]
    except FileNotFoundError:
        return []


# Carica il README in sicurezza
try:
    long_description = Path("README.md").read_text(encoding="utf-8")
except FileNotFoundError:
    long_description = ""


# Mantiene setup.py allineato alle dipendenze runtime del pacchetto
install_requires = read_requirements("requirements/runtime.txt") or ["numpy"]
extras_require = {
    "test": read_requirements("requirements/test.txt"),
    "notebook": read_requirements("requirements/notebook.txt"),
}

setup(
    name="zeldareco",
    version="0.2.0",
    author="Edoardo Maragliano",
    author_email="edoardo.maragliano@edu.unige.it",
    description="A simple package to perform Zeldovich reconstruction of density and velocity fields.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EdoardoMaragliano/ZeldovichReconstruction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require=extras_require,
)
