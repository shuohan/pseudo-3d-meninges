from setuptools import setup

version = '0.1.0'

with open('README.md') as readme:
    long_desc = readme.read()

setup(
    name='pseudo-3d-meninges',
    description='Pseudo-3D Meninges Surface Reconstruction',
    author='Shuo Han',
    author_email='shan50@jhu.edu',
    version=version,
    packages=['deep_meninges'],
    license='GPLv3',
    python_requires='>=3.9.4',
    scripts=['scripts/test.py', 'scripts/train.py', 'scripts/tpc.py'],
    long_description=long_desc,
    install_requires=[
        'torch>=1.8.1',
        'kornia',
        'numpy',
        'nibabel',
        'matplotlib',
        'opencv-python',
        'improc3d',
        'resize@git+https://gitlab.com/shan-utils/resize.git@0.1.1',
        'pytorch-unet@git+https://github.com/shuohan/pytorch-unet.git@0.1.0',
        'ptxl@git+https://gitlab.com/shan-deep-networks/ptxl.git@0.3.1'
    ],
    long_description_content_type='text/markdown',
    url='https://github.com/shuohan/pseudo-3d-meninges',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent'
    ]
)
