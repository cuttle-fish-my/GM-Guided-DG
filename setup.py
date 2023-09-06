from setuptools import setup

setup(
    name="GGADG",
    py_modules=["GGADG"],
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
