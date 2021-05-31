#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

echo "Downloading places365 val_256"
pushd datasets
mkdir -p places365
pushd places365
wget --progress=bar \
   http://data.csail.mit.edu/places/places365/val_256.tar \
   -O val_256.tar
tar -xvf val_256.tar
rm val_256.tar
popd
popd
