from setuptools import setup, find_packages

setup(
    name="iLoco",
    packages=[
        package for package in find_packages() if package.startswith("iLoco")
    ],
)
