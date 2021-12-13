from setuptools import setup

with open("README.md") as f:
    README = f.read()

setup(
    name="dfhist",
    version="0.0.0.post2",
    packages=["dfhist"],
    url="https://github.com/jftsang/dfhist",
    license="CC BY 4.0",
    author="J. M. F. Tsang",
    author_email="j.m.f.tsang@cantab.net",
    description="utilities for caching pandas dataframes",
    setup_requires=[
        "pandas",
    ],
    long_description_content_type='text/markdown',
    long_description=README,
)
