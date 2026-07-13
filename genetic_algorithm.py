import random
from math import atan2, cos, radians, sin, sqrt

import pandas as pd

try:
    from ml_models import encoder, lr_model

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from pathfinding import PathFinder

    PATHFINDING_AVAILABLE = True
except ImportError:
    PATHFINDING_AVAILABLE = False
    print("Warning: pathfinding.py not found. Reverting to straight-line distance.")


class RouteOptimizerGA:
    def __init__(
        self, locations, population_size=50, mutation_rate=0.01, generations=100
    ):
        self.locations = locations
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = []

        if PATHFINDING_AVAILABLE:
            lats = [loc[0] for loc in locations]
            lons = [loc[1] for loc in locations]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)

            max_lat_diff = max(abs(lat - center_lat) for lat in lats)
            max_lon_diff = max(abs(lon - center_lon) for lon in lons)
            radius = max(max_lat_diff, max_lon_diff) * 111000 + 2000

            self.pathfinder = PathFinder(
                center_lat=center_lat, center_lon=center_lon, dist=radius
            )

        self.cost_matrix = self._build_cost_matrix()

    def _haversine(self, coords1, coords2):
        """Fallback geometry calculation if A* fails or isn't available."""
        R = 6371000
        lat1, lon1 = radians(coords1[0]), radians(coords1[1])
        lat2, lon2 = radians(coords2[0]), radians(coords2[1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def _build_cost_matrix(self):
        """Pre-calculates A* distance and ML time between all pairs."""
        n = len(self.locations)
        matrix = {}

        ml_inputs = []
        pairs_map = []

        print("Building Cost Matrix (Calculating Paths)...")

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[(i, j)] = (0, 0)
                    continue

                if PATHFINDING_AVAILABLE:
                    dist_meters = self.pathfinder.get_astar_distance(
                        self.locations[i], self.locations[j]
                    )
                else:
                    dist_meters = self._haversine(self.locations[i], self.locations[j])

                dist_miles = dist_meters / 1609.34

                ml_inputs.append(
                    {
                        "MILES": dist_miles,
                        "START_HOUR": 9,
                        "DAY_OF_WEEK": 0,
                        "CATEGORY": "Business",
                        "PURPOSE": "Unknown",
                        "START_LOC": "Other",
                        "STOP_LOC": "Other",
                    }
                )
                pairs_map.append((i, j, dist_meters))

        if ML_AVAILABLE and len(ml_inputs) > 0:
            input_df = pd.DataFrame(ml_inputs)

            categorical_cols = ["CATEGORY", "PURPOSE", "START_LOC", "STOP_LOC"]
            encoded_cats = encoder.transform(input_df[categorical_cols])
            encoded_df = pd.DataFrame(
                encoded_cats, columns=encoder.get_feature_names_out()
            )

            X_numeric = input_df[["MILES", "START_HOUR", "DAY_OF_WEEK"]]
            X_final = pd.concat([X_numeric, encoded_df], axis=1)

            predicted_times = lr_model.predict(X_final)
        else:
            predicted_times = [x["MILES"] * 2.5 for x in ml_inputs]

        for k, (i, j, dist) in enumerate(pairs_map):
            matrix[(i, j)] = (dist, predicted_times[k])

        return matrix

    def create_individual(self):
        """Creating random permutation (Depot -> Random Stops -> Depot)."""
        indices = list(range(1, len(self.locations)))
        random.shuffle(indices)
        return [0] + indices + [0]

    def initialize_population(self):
        self.population = [self.create_individual() for _ in range(self.pop_size)]

    def fitness(self, individual):
        """Calculates total cost using the pre-calculated matrix."""
        total_distance = 0
        total_time = 0

        for i in range(len(individual) - 1):
            u, v = individual[i], individual[i + 1]
            d, t = self.cost_matrix.get((u, v), (0, 0))
            total_distance += d
            total_time += t

        return 1 / (total_distance + 1e-5)

    def selection(self):
        """Tournament Selection."""
        k = 3
        selected = []
        for _ in range(self.pop_size):
            aspirants = random.choices(self.population, k=k)
            best = max(aspirants, key=self.fitness)
            selected.append(best)
        return selected

    def crossover(self, parent1, parent2):
        """Ordered Crossover (OX1)."""
        size = len(parent1) - 2
        if size < 1:
            return parent1

        start, end = sorted(random.sample(range(1, size + 1), 2))

        child = [None] * len(parent1)
        child[0], child[-1] = 0, 0
        child[start:end] = parent1[start:end]

        p2_genes = [item for item in parent2[1:-1] if item not in child]
        pointer = 0
        for i in range(1, len(child) - 1):
            if child[i] is None:
                child[i] = p2_genes[pointer]
                pointer += 1
        return child

    def mutate(self, individual):
        """Swap Mutation."""
        if len(individual) > 3 and random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(1, len(individual) - 1), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    def run(self):
        print(f"Initializing GA optimization for {len(self.locations)} locations...")
        self.initialize_population()

        global_best_route = None
        global_best_fitness = -float("inf")

        for gen in range(self.generations):
            current_best = max(self.population, key=self.fitness)
            current_best_fitness = self.fitness(current_best)

            if current_best_fitness > global_best_fitness:
                global_best_fitness = current_best_fitness
                global_best_route = current_best

            next_pop = [global_best_route]

            parents = self.selection()

            while len(next_pop) < self.pop_size:
                p1, p2 = random.sample(parents, 2)
                child1 = self.crossover(p1, p2)
                self.mutate(child1)
                next_pop.append(child1)

                if len(next_pop) < self.pop_size:
                    child2 = self.crossover(p2, p1)
                    self.mutate(child2)
                    next_pop.append(child2)

            self.population = next_pop

            if gen % 20 == 0:
                print(f"Generation {gen}: Best Fitness = {global_best_fitness:.5f}")

        return global_best_route
