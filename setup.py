from setuptools import setup, find_packages

HYPHEN_E_DOT = "-e ."

def get_requirements(filename):
    """Reads the requirements from a file and returns a list of requirements."""

    with open(filename) as file_object:
        requirements = [line.strip() for line in file_object 
                if line.strip() and not line.startswith('#')]

    # Remove the editable install requirement if present
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="ML_Project",
    version="0.1.0",
    author="Aneesh Jose",
    author_email="aneeshjose012@gmail.com",
    description="A machine learning project",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)