from setuptools import setup, find_packages

setup(
    name="iSLAM",
    packages=[
        package for package in find_packages() if package.startswith("iSLAM")
    ],
)
