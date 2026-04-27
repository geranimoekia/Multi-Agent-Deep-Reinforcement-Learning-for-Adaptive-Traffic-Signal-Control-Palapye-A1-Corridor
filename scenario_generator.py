import random

class ScenarioGenerator:
    """
    Controls traffic demand + randomness for SUMO + PPO training
    """

    def __init__(self):
        self.step = 0
        self.demand = 1.0

        # randomness parameters
        self.spawn_noise = 0.3
        self.route_bias = 0.5

    def update(self, step):
        self.step = step

        # Curriculum-style demand shaping
        if step < 1000:
            self.demand = 0.5
        elif step < 3000:
            self.demand = 1.2
        else:
            self.demand = 2.0

    def sample_route_bias(self):
        return random.uniform(0.3, 1.0) * self.demand

    def apply_traci(self, traci):
        """
        Inject vehicles dynamically (simple template)
        Expand this later with real OD logic
        """

        try:
            if self.step % max(1, int(10 / self.demand)) == 0:

                veh_id = f"veh_{self.step}_{random.randint(0,9999)}"

                edges = traci.edge.getIDList()
                if len(edges) < 2:
                    return

                from_edge = random.choice(edges)
                to_edge = random.choice(edges)

                route_id = f"route_{veh_id}"

                traci.route.add(route_id, [from_edge, to_edge])
                traci.vehicle.add(veh_id, route_id, typeID="car_normal")

        except Exception:
            pass