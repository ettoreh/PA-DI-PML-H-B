from setuptools import setup, find_packages

setup(
    name='pml',
    version='0.1.0',
    packages=find_packages(
        include=['content', 'content.*',
                 'dataset_loader', 'dataset_loader.*',
                 'content.watermark', 'content.watermark.*', 'networks', 'networks.*']),
    install_requires=[]
)