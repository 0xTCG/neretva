#!/bin/bash

# SCRIPT=$1
# SUFFIX=$2

PREFIX="results_CYP2D6_new"
mkdir -p "$PREFIX"

for gene in CYP2D6; do
# for gene in CYP2B6 CYP4F2 CYP3A5; do
# for gene in CYP4F2 TPMT CYP3A5 SLCO1B1; do
# for gene in CYP4F2; do
    mkdir -p "${PREFIX}/${gene}"
    ls /project/shared/aldy-data/wgs/*.wgs.cram | \
    /cvmfs/soft.computecanada.ca/gentoo/2020/usr/bin/parallel -j6 \
    "id=\$(basename {} .wgs.cram); \
     [ -f ${PREFIX}/${gene}/\${id}.log ] && echo \"Skip \${id}\" || \
     /cvmfs/soft.computecanada.ca/gentoo/2020/usr/bin/time -v python cyp.py ${gene} {} > ${PREFIX}/${gene}/\${id}.log 2>&1"
done
