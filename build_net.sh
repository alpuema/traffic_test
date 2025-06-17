#!/bin/bash
netgenerate --grid --grid.number=2 --output-file=net/grid.net.xml
python $SUMO_HOME/tools/randomTrips.py -n net/grid.net.xml -o routes/trips.trips.xml --seed 42 --trip-attributes="departLane=\"best\" departSpeed=\"max\" departPos=\"random\""
duarouter -n net/grid.net.xml -t routes/trips.trips.xml -o routes/routes.rou.xml
echo "Network and routes generated." 