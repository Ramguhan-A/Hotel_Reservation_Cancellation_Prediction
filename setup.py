from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    

setup(
    name = "Hotel_reversation_prediction_MLops",
    description= "This project aims to predict which hotel reservation booking are likely to be canceled. so, that we can help hotels to identify them and plan accordingly to reduce revenue loss, target marketing etc..",
    author= "Ramguhan",
    version= "0.1",
    packages= find_packages(),
    install_requires = requirements,
)