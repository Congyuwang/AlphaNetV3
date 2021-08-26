from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='alphanet',
    version='0.0.19',
    packages=['alphanet'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Congyuwang/AlphaNetV3',
    license='MIT License',
    author='Congyu Wang',
    author_email='leonwang998@foxmail.com',
    description='A recurrent neural network for predicting '
                'stock market performance',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=[]
)
