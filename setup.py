from setuptools import find_packages, setup
from typing import List

hyphen_e_dot = "-e ."

def get_requirements(file_path:str) -> List[str]:
    """
    This function returns the list of requirements from the file path passed as a parameter
    """
    requirements=[]

    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)

    return requirements

setup(
    name="Data Science Project",
    version="0.0.1",
    author="Prathamesh",
    author_email="prathameshj7580@gmail.com",
    packages=find_packages(),
install_requires=get_requirements("requirements.txt")
)