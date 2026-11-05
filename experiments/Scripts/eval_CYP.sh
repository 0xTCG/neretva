#!/bin/bash

PREFIX=${1:-results_CYP_WGS}
DATA=${DATA:-/project/shared/aldy-data/wgs}
REF=${REF:?set REF to the GRCh37 reference FASTA the CRAMs were converted against}
PKG=${PKG:-$(cd "$(dirname "$0")/../.." && pwd)}
JOBS=${JOBS:-6}

export PYTHONPATH=$PKG:$PKG/neretva
cd $PKG/neretva

mkdir -p "$PREFIX"

for gene in CYP2D6; do
    mkdir -p "${PREFIX}/${gene}"
    ls $DATA/*.wgs.cram | parallel -j $JOBS \
    "id=\$(basename {} .wgs.cram); python cyp.py --gene ${gene} --input {} --reference $REF --seed 42 > ${PREFIX}/${gene}/\${id}.log 2>&1"
done
