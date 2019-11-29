import setuptools
from io import open
import os

version_module = {}
dirname = os.path.dirname(__file__)
with open(os.path.join(dirname, "src/pasio/version.py")) as fp:
    exec(fp.read(), version_module)
    __version__ = version_module['__version__']

with open(os.path.join(dirname, "README.md"), encoding='utf8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="pasio", # Replace with your own username
    version=__version__,
    author="Andrey Lando, Ilya Vorontsov",
    author_email="dronte.l@gmail.com, vorontsov.i.e@gmail.com",
    description="Pasio is a tool for segmentation and denosing DNA coverage profiles coming from high-throughput sequencing data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autosome-ru/pasio",
    license="Pasio is licensed under WTFPL, but if you prefer more standard licenses, feel free to treat it as MIT license",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    keywords='bioinformatics NGS coverage segmentation denoise',
    classifiers=[
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=2.7.1, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*',
    install_requires=['numpy >= 1.8.0', 'scipy>=0.12.0', 'future >= 0.4.0',],
    extras_require={
        'dev': ['pytest', 'pytest-benchmark', 'flake8', 'tox', 'wheel', 'twine', 'setuptools_scm'],
    },
    entry_points={
        'console_scripts': [
            'pasio=pasio.cli:main',
        ],
    },
    use_scm_version=False,
    setup_requires=['setuptools_scm'],
)
