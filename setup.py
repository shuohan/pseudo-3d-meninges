from setuptools import setup

version = '0.2.0'

with open('README.md') as readme:
    long_desc = readme.read()

setup(
    name='ct_synth',
    description='Mallika\'s CT synthesis.',
    author='Shuo Han',
    author_email='shan50@jhu.edu',
    version=version,
    packages=['ct_synth'],
    license='GPLv3',
    python_requires='>=3.7.10',
    scripts=['scripts/test.py', 'scripts/train.py'],
    long_description=long_desc,
    install_requires=[
        'torch>=1.8.1',
        'numpy',
        'nibabel',
        'opencv-python',
        'pytorch-unet@git+https://github.com/shuohan/pytorch-unet.git@0.1.0',
        'ptxl@git+https://gitlab.com/shan-deep-networks/ptxl.git@0.3.1'
    ],
    long_description_content_type='text/markdown',
    url='https://gitlab.com/iacl/mallika-ct-synth.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent'
    ]
)
