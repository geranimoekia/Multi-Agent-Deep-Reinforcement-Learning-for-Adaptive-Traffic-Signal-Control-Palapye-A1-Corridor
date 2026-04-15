import traci
import asyncio
import websockets
import json

SUMO_PORT = 8813

async def stream_data(websocket):
    traci.init(SUMO_PORT)

    while True:
        traci.simulationStep()

        vehicles = traci.vehicle.getIDList()
        output = []

        for v in vehicles:
            x, y = traci.vehicle.getPosition(v)
            angle = traci.vehicle.getAngle(v)

            output.append({
                "id": v,
                "x": x,
                "y": y,
                "angle": angle
            })

        await websocket.send(json.dumps(output))
        await asyncio.sleep(0.1)

async def main():
    async with websockets.serve(stream_data, "localhost", 8765):
        print("WebSocket running on ws://localhost:8765")
        await asyncio.Future()

asyncio.run(main())