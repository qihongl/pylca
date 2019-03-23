from setuptools import setup, find_packages

setup(name='pylca',
      version='0.52',
      description='a package for the leaky competing accumulator',
      keywords='cognitive science',
      url='https://github.com/qihongl/pylca',
      author='Qihong Lu',
      author_email='lvqihong1992@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
      ],
      zip_safe=False)
