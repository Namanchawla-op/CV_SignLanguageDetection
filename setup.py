import pathlib
import setuptools
import pkg_resources

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text(encoding='utf-8')

with pathlib.Path('docs/requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

# Read meta-data
about = {}
exec(open('openhands/__version.py').read(), about)

# Install
setuptools.setup(
    name="OpenHands",
    version=about["__version__"],
    description="👐OpenHands : Making Sign Language Recognition Accessible",
    long_description=README,
    long_description_content_type="text/markdown",
    download_url="https://pypi.org/project/OpenHands",
    author="",
    # packages=["openhands"],
    packages=setuptools.find_packages(),
    # packages=setuptools.find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries"
    ],
)
