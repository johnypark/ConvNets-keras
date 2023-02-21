import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ConvNets-keras", # Replace with your own username
    version="1.0",
    author="John Park",
    author_email="parkjohnyc@gmail.com",
    description="Modern CNNs in keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnypark/ConvNets-keras",
    packages=setuptools.find_packages(),
    install_requires = ['tensorflow',
                       'tensorflow-addons',
                        'typeguard'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

