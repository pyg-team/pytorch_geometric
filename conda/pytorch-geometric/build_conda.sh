#!/bin/bash

conda build . -c defaults -c pytorch -c rusty1s --output-folder "$HOME/conda-bld"
