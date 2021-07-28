#!/bin/bash

cd src || exit
pdoc -o ../doc -d google alphanet alphanet.data alphanet.metrics
