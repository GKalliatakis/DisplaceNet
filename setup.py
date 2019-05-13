from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name='DisplaceNet',
    version='0.1',
    author="Grigorios Kalliatakis",
    author_email="gkallia@essex.ac.uk",
    description="Recognising Displaced People from Images by Exploiting Dominance Level",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GKalliatakis/DisplaceNet",
    download_url="https://github.com/GKalliatakis/DisplaceNet/archive/master.zip",
    licence="MIT",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ],
)


