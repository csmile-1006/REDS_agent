import glob
import pathlib
import setuptools
from setuptools import find_namespace_packages


setuptools.setup(
    name='reds_rl',
    version='0.0.3',
    description='Subtask-Aware Visual Reward Learning from Segmented Demonstrations',
    author='Changyeon Kim',
    url='http://github.com/csmile-1006/REDS_reward_learning',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=find_namespace_packages(exclude=['scripts', 'reds_rl_data.datasets', 'reds_rl_data.datasets.*']),
    include_package_data=True,
    scripts=glob.glob('reds_rl_data/download/*.sh'),
    install_requires=pathlib.Path('requirements.txt').read_text().splitlines(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)