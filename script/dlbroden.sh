#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

# Download broden1_224
if [ ! -f datasets/broden1_224/index.csv ]
then

echo "Downloading broden1_224"
mkdir -p datasets
pushd datasets
wget --progress=bar \
   http://netdissect.csail.mit.edu/data/broden1_224.zip \
   -O broden1_224.zip
#cp ~/broden1_224.zip .
unzip -q broden1_224.zip
rm broden1_224.zip
popd

fi
