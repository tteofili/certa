import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CERTA",
    version="0.0.2",
    author="Tommaso Teofili",
    author_email="tommaso.teofili@gmail.com",
    description="Computing ER explanations with TriAngles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url= 'https://github.com/tteofili/certa.git',
    packages=['certa'],
    install_requires=[
          'pandas',
          'numpy',
          'scipy',
          'scikit-learn',
          'tqdm',
          'transformers',
          'torch',
          'tensorflow',
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
)
