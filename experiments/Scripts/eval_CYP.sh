#!/bin/bash

# SCRIPT=$1
# SUFFIX=$2

PREFIX="results_CYP_WGS_0511"
mkdir -p "$PREFIX"

for gene in CYP2D6; do
# for gene in CYP2C8 CYP2C9 CYP2C19 ; do

    mkdir -p "${PREFIX}/${gene}"
    ls /project/shared/aldy-data/wgs/*.wgs.cram | \
    /cvmfs/soft.computecanada.ca/gentoo/2020/usr/bin/parallel -j6 \
    "id=\$(basename {} .wgs.cram); /cvmfs/soft.computecanada.ca/gentoo/2020/usr/bin/time -v python cyp.py ${gene} {} > ${PREFIX}/${gene}/\${id}.log 2>&1"
done