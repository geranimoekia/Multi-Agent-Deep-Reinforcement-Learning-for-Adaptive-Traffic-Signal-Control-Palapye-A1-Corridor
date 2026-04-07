#!/bin/bash
# Run this from the folder containing network_tripled.net.xml
# Requires SUMO tools on your PATH (or adjust SUMO_HOME)

IN_EDGES="-465932558#2_C 470773638#0_C -E5_B 470773638#0_B -E5 470773638#0 -470773638#0 E0"
OUT_EDGES="465932558#2_C -470773638#0_C E5_B -470773638#0_B E5 E6"

python3 $SUMO_HOME/tools/randomTrips.py \
  --net-file network_tripled.net.xml \
  --output-trip-file trips_triple.trips.xml \
  --route-file triple_routes_clean.rou.xml \
  --begin 0 \
  --end 5000 \
  --period 2 \
  --fringe-factor 1 \
  --from-edges "$IN_EDGES" \
  --to-edges "$OUT_EDGES"

echo "Done. Route file: triple_routes_clean.rou.xml"
