from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torchnise",
    version="0.3.3",
    author="Yannick Holkamp, Emiliano Godinez, Ulrich Kleinekathöfer",
    author_email="yholtkamp@constructor.university",
    description="several nise extensions. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ausstein/MLNISE",
    packages=find_packages(),  # Automatically find packages in the project
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Appache",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "torch>=2.2.2",
        "tqdm",
        "h5py"
    ],
)