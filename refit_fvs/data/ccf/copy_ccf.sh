#!/bin/bash

variants="ak bm ca ec pn wc so ws cr ie nc"
for v in $variants; do
  cp /mnt/e/ecotrust/github/ForestVegetationSimulator/${v}/ccfcal.f ./ccfcal_${v}.f
done

