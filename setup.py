from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torchnise",
    version="0.3.0",
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
        "matplotlib~=3.9",
        "numpy==1.26.4",
        "scipy==1.14.0",
        "torch==2.2.2",
        "torchvision==0.17.2",
        "tqdm==4.66.4",
        "h5py"
    ],
)