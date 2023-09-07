from setuptools import setup

setup(
    name="GMGADG",
    py_modules=["GMGADG"],
    install_requires=[
        "matplotlib",
        "numpy",
        "torch",
        "torchvision",
        "einops",
        "tqdm",
        "blobfile",
        "monai",
        "nibabel",
        "tensorboard"
    ],
)
