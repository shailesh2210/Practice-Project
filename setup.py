from setuptools import setup , find_packages
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str) -> List[str]:
    """
    This Function will return the list of requirements
    """
    requirements = []
    with open (file_path) as file_obj:
        requirements = file_obj.readline()  #reading the lines
        requirements = [req.replace("\n", "") for req in requirements]  #replace slash n with blank space

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(name = "Ipl Score Prediction",
version = 0.0.1,
author="Shailesh Gaddam"
author_email = "shaileshgaddam22@gmail.com",
packages = find_packages()
install_requires= get_requirements("requirements.txt")
)