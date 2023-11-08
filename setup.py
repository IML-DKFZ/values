import os
from setuptools import setup, find_packages


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1])
                )
            else:
                requirements.append(r)
    return requirements


requirements = resolve_requirements(
    os.path.join(os.path.dirname(__file__), "requirements.txt")
)


setup(
    name="uncertainty_modeling",
    version="0.1",
    packages=find_packages(),
    python_requires="==3.10",
    install_requires=requirements,
    description="Uncertainty Modeling for Semantic Segmentation",
)
