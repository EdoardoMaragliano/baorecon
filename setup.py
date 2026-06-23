from pathlib import Path

from setuptools import setup, find_packages



def read_requirements(path):
    req_file = Path(path)
    if not req_file.exists():
        return []
    
    # Filtriamo solo le righe che sono valide dipendenze
    lines = req_file.read_text(encoding="utf-8").splitlines()
    requirements = []
    for line in lines:
        line = line.strip()
        # Ignora commenti, righe vuote e inclusioni di altri file (che pip gestisce internamente)
        if line and not line.startswith("#") and not line.startswith("-r"):
            requirements.append(line)
    return requirements


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
    author_email="edoardo.maragliano@gmail.com",
    description="A simple package to perform Zeldovich reconstruction of density and velocity fields.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EdoardoMaragliano/baorecon",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
)
