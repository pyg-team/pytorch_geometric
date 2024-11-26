#!/bin/sh

# Check if the user provided a Python file
if [ -z "$1" ]; then
    echo "Usage: $0 <python_file>"
    exit 1
fi

# Check if the provided file exists
if [[ ! -f "$1" ]]; then
    echo "Error: File '$1' does not exist."
    exit 1
fi

# Check if the provided file is a Python file
if [[ ! "$1" == *.py ]]; then
    echo "Error: '$1' is not a Python file."
    exit 1
fi

# Get the base name of the Python file
python_file=$(basename "$1")

# Run nsys profile on the Python file
nsys profile -c cudaProfilerApi --capture-range-end repeat -t cuda,nvtx,osrt,cudnn,cublas --cuda-memory-usage true --cudabacktrace all --force-overwrite true --output=profile_${python_file%.py} python "$1"

echo "Profile data saved as profile_${python_file%.py}.nsys-rep"
