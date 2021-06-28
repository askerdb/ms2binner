import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ms2binner",
    version="1.0.2",
    author="Asker Brejnrod & Arjun Sampath",
    description="Filters and bins ms2 spectra",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/askerdb/ms2binner",
    license="Apache Software License 2.0",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    extras_require={"dev": ["sphinxcontrib-apidoc",
			     "pytest"]
    }
)
