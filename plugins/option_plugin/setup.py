from setuptools import setup, find_packages

setup(
    name='gquant_option_plugin',
    packages=find_packages(include=['gquant_option_plugin']),
    entry_points={
        'gquant.plugin':
        ['gquant_option_plugin = gquant_option_plugin']
    }
)
