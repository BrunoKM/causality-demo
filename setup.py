import setuptools

setuptools.setup(
    name="causality-demos",
    version="0.0.1",
    author="Bruno Kacper Mlodozeniec",
    author_email="bkmlodozeniec@gmail.com",
    description="Demos demonstrating links between causality and probabilistic inference",
    long_description=open("README.md", 'r').read(),
    url="https://github.com/brunokm/causality-demo",
    packages=['causalitydemos'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)