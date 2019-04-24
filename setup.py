from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.0.1',
    description='Our goal is to utilize deep learning algorithms to perform binary classification on MRI images to detect the presence or absence of a brain tumor. As an extended/secondary goal, we also hope to perform segmentation and identify tumorous pixels in MRI images. Our dataset, found on Kaggle (Link), contains 253 MRI scans of the human brain, broken into two classes, 155 tumorous scans and 98 non-tumorous scans.',
    author='Garrett Eickelberg, Sabarish Chockalingam,Andrew Hall, Naomi Kaduwela, Matthew Kehoe, Molly Srour',
    license='MIT',
)
