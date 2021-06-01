import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

REQUIRED = [
    'numpy~=1.0',
    'scikit-learn~=0.24',
    'pandas~=1.0',
    'rpy2~=3.0'
]

SETUP_REQUIRED = [
    'setuptools_scm'
]

setuptools.setup(
    name="EBLR",
    version="0.0.1",
    author="Igor Ilic",
    author_email="iilic@ryerson.ca",
    description="Explainable Boosted Linear Regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/irogi/eblr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude="tests"),
    install_requires=REQUIRED,
    setup_requires=SETUP_REQUIRED,
    python_requires=">=3.6"
)