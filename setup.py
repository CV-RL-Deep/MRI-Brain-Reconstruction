import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# reading requirments.txt file
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setuptools.setup(
    name="brec-mri-reconstruction",  # pip project name
    version="0.1.0",
    author="Matvey Faro",
    author_email="m.czygankov@bk.ru",
    description="Brain MRI Reconstruction pipeline (2.5D SPADE AR)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matveyfaro/MRI-Brain-Reconstruction",
    packages=setuptools.find_packages(where="experiments"),
    package_dir={"": "experiments"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
    install_requires=requirements,
)
