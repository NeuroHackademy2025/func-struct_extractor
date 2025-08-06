from setuptools import setup, find_packages

setup(
    name="func-struct_extractor",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "click",
        "numpy",
        "pandas",
        "dipy",
        "matplotlib",
        "nilearn",
        "pyyaml"
    ],
    entry_points={
        "console_scripts": [
            "funcstruct-extractor=func_struct_extractor.app:cli"
        ]
    },
)
