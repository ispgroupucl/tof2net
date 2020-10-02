from setuptools import setup, find_packages

setup(name="tofnet",
      version="3.2",
      description="TofNet models, utilities and extras",
      url="http://github.com/ispgroupucl/tof2net",
      author="ISPGroup",
      author_email="victor.joos@uclouvain.be",
      scripts=["tof"],
      packages=find_packages(),
      zip_safe=False
)