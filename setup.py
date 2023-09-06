from setuptools import setup

setup(
    name="GGADG",
    py_modules=["GGADG"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)