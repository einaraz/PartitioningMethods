from setuptools import setup, find_packages

setup(
    name='partitioning',
    version='1.0.1',
    packages=find_packages(),  # Automatically find and include all packages
    author='Einara Zahn',
    author_email='einara.zahn@gmail.com',  # Your email
    url='https://github.com/einaraz/PartitioningMethods',
    install_requires=[
        'matplotlib==3.8.0',
        'numpy==2.0.1',
        'pandas==2.2.2',
        'Pint==0.24.3',
        'pytest==7.4.0',
        'scipy==1.14.0'
    ,]
)
