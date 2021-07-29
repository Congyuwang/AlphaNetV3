#!/bin/bash

pydocstyle src/
lazydocs --overview-file="README.md" src/ --output-path ./docs/
cd src || exit
pdoc -o ../docs/html/ -d google alphanet alphanet.data alphanet.metrics
