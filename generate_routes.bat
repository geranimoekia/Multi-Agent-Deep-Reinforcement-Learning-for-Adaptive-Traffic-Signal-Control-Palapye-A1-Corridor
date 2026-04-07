@echo off
REM ============================================================
REM  Run this from C:\DRL_TRAFFIC\
REM  Requires SUMO_HOME to be set (usually done by SUMO installer)
REM ============================================================

set NET=network_tripled.net.xml
set TRIPS=trips_triple.trips.xml
set ROUTES=triple_routes_clean.rou.xml

set IN_EDGES=-465932558#2_C 470773638#0_C -E5_B 470773638#0_B -E5 470773638#0 -470773638#0 E0
set OUT_EDGES=465932558#2_C -470773638#0_C E5_B -470773638#0_B E5 E6

echo Generating trips with randomTrips.py...

python "%SUMO_HOME%\tools\randomTrips.py" ^
  --net-file %NET% ^
  --output-trip-file %TRIPS% ^
  --route-file %ROUTES% ^
  --begin 0 ^
  --end 5000 ^
  --period 2 ^
  --fringe-factor 1 ^
  --from-edges "%IN_EDGES%" ^
  --to-edges "%OUT_EDGES%" ^
  --ignore-errors ^
  --verbose

echo.
echo Done! Route file: %ROUTES%
echo Now run: sumo-gui -c triple.sumocfg
pause
