#!/bin/bash

conda build . -c defaults -c pytorch -c conda-forge -c rusty1s --output-folder "$HOME/conda-bld"
