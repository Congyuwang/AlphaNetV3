#!/bin/bash

lazydocs --overview-file="README.md" src/ --output-path ./docs/md/
pydocstyle src/
cd src || exit
pdoc -o ../docs/html/ -d google alphanet alphanet.data alphanet.metrics
