import random
import time
import timeit
import statistics
import matplotlib.pyplot as plt
SHOW_PLOTS = False   # Global flag – set True to enable any plotting
import math  # Required for entropy calculations
import json
import numpy as np
import copy
import sys
from pathlib import Path  # needed for TSPLIB loader

# For optional caching (used in DTSP novelty, if needed)
from functools import lru_cache

# Global GA parameters
GA_POPSIZE = 300    # Population size for the genetic algorithm
GA_MAXITER = 16384    # Maximum number of generations (used in non-Bin Packing modes)
# Stop if the global best fitness hasn't improved for this many generations
GA_NO_IMPROVEMENT_LIMIT = 300
GA_ELITRATE = 0.1                # Elitism rate (percentage of best individuals preserved)
GA_MUTATIONRATE = 0.25            # Mutation probability

# ───── Mutation–control policies (Task 2‑a) ────────────────────────── #
GA_MUTATION_POLICY  = "CONSTANT"   # "CONSTANT", "LOGISTIC", "THM"
# LOGISTIC‑decay parameters
GA_MUTRATE_MAX      = 0.9
GA_MUTRATE_MIN      = 0.1
GA_DECAY_RATE       = 0.01         # decay coefficient r
# Triggered‑Hyper‑Mutation (THM) parameters
GA_MUTRATE_BASE     = 0.25
GA_MUTRATE_HYPER    = 0.9
GA_THM_PATIENCE     = 20           # gens without improvement ⇒ burst
GA_THM_DURATION     = 40            # gens to keep hyper‑µ active
# runtime variables (updated each generation)
GA_DYNAMIC_MUTRATE  = GA_MUTATIONRATE   # effective µ for the current gen
_THM_BURST_REMAINING = 0
# --- logging helper: average individual‑level mutation probability ---
GEN_AVG_INDIV_MUT_HISTORY: list[float] = []
# ───────────────────────────────────────────────────────────────────── #


# ───── Individual‑level adaptive mutation (Task 2‑b) ───── #
# Options: "NONE" (default), "FITNESS" (relative fitness), "AGE" (age‑based)
GA_INDIV_MUT_POLICY = "NONE"
_INDIV_MUT_MIN      = GA_MUTRATE_MIN
_INDIV_MUT_MAX      = GA_MUTRATE_MAX
# --------------------------------------------------------- #


# ───── Adaptive *fitness* augmentation (Task 2‑c) ───── #
# Choose an auxiliary reward g(x,t) that is **added** (or subtracted)
# to the raw problem fitness before selection is applied.
# Options: "NONE" (raw fitness only), "AGE" (age‑based reward),
#          "NOVELTY" (k‑NN behavioural‑novelty reward)
GA_ADAPT_FIT_POLICY = "NONE"
AGE_WEIGHT          = 1.0       # fitness += AGE_WEIGHT * age
NOVELTY_K           = 3         # number of neighbours for novelty
NOVELTY_WEIGHT      = 10       # fitness -= NOVELTY_WEIGHT * novelty
GEN_AVG_G_HISTORY: list[float] = []   # log ⟨g(x,t)⟩ per generation

# ───── Niching / Speciation (Task 4) ───── #
# Choose how to reduce crowding in the population.
#   "NONE"       : no niching (default – legacy behaviour)
#   "SHARING"    : fitness‑sharing within a radius (Goldberg & Richardson, 1987)
#   "SPECIATION" : NEAT‑style speciation threshold
GA_NICHING_ALGO  = "NONE"     #  "NONE" | "SHARING" | "SPECIATION"
SHARING_RADIUS   = 0.4       # r_s  – neighbourhood radius for fitness‑sharing
SPECIES_THRESHOLD = 0.30      # δ_t – max intra‑species distance
SPECIES_PENALTY  = 0.10       # additive penalty per extra member in a species

# ─── Performance tweak: compute novelty only every k generations ─── #
NOVELTY_PERIOD      = 5      # calculate novelty reward once every k gens
CURRENT_GENERATION  = 0      # updated from the evolution loops
# ─────────────────────────────────────────────────────────────────── #

# Novelty sampling: sample size for estimating novelty (speeds‑up)
NOVELTY_SAMPLE = 25        # sample size when estimating novelty (speeds‑up)

GA_TARGET = "Hello World!"        # Target string for the String evolution mode
GA_CROSSOVER_OPERATOR = "SINGLE"  # Default crossover operator; may be updated based on user input

# Global fitness heuristic parameters
GA_FITNESS_HEURISTIC = "ORIGINAL"  # Fitness heuristic ("ORIGINAL" or "LCS")
GA_BONUS_FACTOR = 0.5              # Bonus factor for correct positions in LCS-based fitness

# ---------- Task 10: Parent Selection Method Parameters ----------
# Parent selection methods: "RWS", "SUS", "TournamentDet", "TournamentStoch", "Original"
GA_PARENT_SELECTION_METHOD = "RWS"  # Default parent selection method (can be updated by user)
GA_TOURNAMENT_K = 5                # Tournament size for tournament-based selection
GA_TOURNAMENT_P = 0.8              # Probability of selecting the best in a stochastic tournament
GA_MAX_AGE = 20                    # Maximum age for an individual (for aging survivor selection)

# Global mode variables; these change the problem being solved.
# Options: "STRING" (for Hello World evolution), "ARC" (for ARC puzzles), or "BINPACKING" (for bin packing problems)
GA_MODE = "STRING"
GA_ARC_TARGET_GRID = None
GA_ARC_INPUT_GRID = None

# Global variables for BINPACKING mode
BP_ITEMS = []      # List of item sizes (integers) for the bin packing problem
BP_CAPACITY = None # Maximum capacity of each bin (e.g., 150)
BP_OPTIMAL = None  # The theoretical optimal number of bins (provided in the file)

# הוספת מצב DTSP
DTSP_CITIES = []         # רשימת קואורדינטות ערים

_DISTANCE_CACHE = {}

# ───────────────────── VALIDATION HELPERS ───────────────────── #
def is_hamiltonian_cycle(path) -> bool:
    """
    Return True iff `path` contains every city exactly once
    and therefore forms a Hamiltonian cycle (closure is implicit
    by wrapping from last➜first).
    """
    n = len(DTSP_CITIES)
    return len(path) == n and len(set(path)) == n and set(path) == set(range(n))

def tours_edge_overlap(path1, path2) -> bool:
    """
    Return True if the undirected edge‑sets of the two paths overlap.
    Each edge is considered without orientation.
    """
    edges1 = {tuple(sorted((path1[i], path1[(i + 1) % len(path1)])))
              for i in range(len(path1))}
    edges2 = {tuple(sorted((path2[i], path2[(i + 1) % len(path2)])))
              for i in range(len(path2))}
    return not edges1.isdisjoint(edges2)

def _dist(a: int, b: int) -> float:
    """
    Symmetric Euclidean distance between city indices a and b with memo‑cache.
    """
    key = (a, b) if a <= b else (b, a)
    if key in _DISTANCE_CACHE:
        return _DISTANCE_CACHE[key]
    x1, y1 = DTSP_CITIES[key[0]]
    x2, y2 = DTSP_CITIES[key[1]]
    d = math.hypot(x2 - x1, y2 - y1)
    _DISTANCE_CACHE[key] = d
    return d

# ───────── Gene‑Distance metrics  (Task 3) ───────── #

_MAX_CITY_DIST = None  # DTSP – calculated lazily once

def gene_distance_dtsp(city_a: int, city_b: int) -> float:
    """
    Gene distance for DTSP.
    A gene ≡ city index; distance is the normalised Euclidean distance
    between the two cities in [0,1].

        0.0 → identical city
        1.0 → farthest‑apart pair of cities in this instance
    """
    if city_a == city_b:
        return 0.0
    global _MAX_CITY_DIST
    if _MAX_CITY_DIST is None:
        n = len(DTSP_CITIES)
        _MAX_CITY_DIST = max(
            _dist(i, j) for i in range(n) for j in range(i + 1, n)
        )
    # fall back to 1 to avoid division‑by‑zero on degenerate instances
    denom = _MAX_CITY_DIST or 1.0
    return _dist(city_a, city_b) / denom


def gene_distance_binpack(item_i: int, item_j: int) -> float:
    """
    Gene distance for Bin‑Packing.
    A gene ≡ item index; distance is the absolute size difference
    normalised by the bin capacity (range [0,1]).

        0.0 → items of identical size
        1.0 → maximal possible difference (== BP_CAPACITY)
    """
    if item_i == item_j:
        return 0.0
    # If BP_CAPACITY is not yet initialised, protect against ZeroDivision
    denom = float(BP_CAPACITY or 1)
    return abs(BP_ITEMS[item_i] - BP_ITEMS[item_j]) / denom

# ─────────────────────────────────────────────────── #

# ───────── dynamic mutation‑rate scheduler ────────────────────────── #
def compute_mutation_rate(generation: int, stagnation_counter: int) -> float:
    """
    Return µ(t) according to GA_MUTATION_POLICY.
    • LOGISTIC : µ = µ_min + (µ_max‑µ_min)·e^(‑r·t)
    • THM      : If stagnation ≥ patience → burst (µ_hyper) for `GA_THM_DURATION`
    • CONSTANT : legacy fixed GA_MUTATIONRATE.
    """
    global _THM_BURST_REMAINING
    if GA_MUTATION_POLICY.upper() == "LOGISTIC":
        return GA_MUTRATE_MIN + (GA_MUTRATE_MAX - GA_MUTRATE_MIN) * math.exp(-GA_DECAY_RATE * generation)
    elif GA_MUTATION_POLICY.upper() == "THM":
        if _THM_BURST_REMAINING > 0:
            _THM_BURST_REMAINING -= 1
            return GA_MUTRATE_HYPER
        if stagnation_counter >= GA_THM_PATIENCE:
            _THM_BURST_REMAINING = GA_THM_DURATION - 1
            return GA_MUTRATE_HYPER
        return GA_MUTRATE_BASE
    else:  # CONSTANT
        return GA_MUTATIONRATE

# ───────── per‑individual adaptive mutation probability ───────── #
def compute_individual_mutrate(parent1, parent2, avg_fit):
    """
    Derive offspring mutation probability according to GA_INDIV_MUT_POLICY,
    **combined multiplicatively** with the *global* population‑level rate
    GA_DYNAMIC_MUTRATE so that both schedulers operate together.

    • "NONE"    : μ_i = GA_DYNAMIC_MUTRATE          (population wide only)
    • "FITNESS" : μ_i = μ_global · (1 – relative_fitness)
    • "AGE"     : μ_i = μ_global · age_norm
                 where age_norm = max(parent.age)/GA_MAX_AGE  in [0,1]
    The returned value is finally clamped to [_INDIV_MUT_MIN, _INDIV_MUT_MAX].
    """
    # always start from global scheduler value
    base_mu = GA_DYNAMIC_MUTRATE

    if GA_INDIV_MUT_POLICY == "FITNESS" and avg_fit > 0:
        rel_fit = ((parent1.fitness + parent2.fitness) / 2) / avg_fit   # ∈ ℝ⁺
        prob = base_mu * (1.0 - rel_fit)                                # worse → higher μ
    elif GA_INDIV_MUT_POLICY == "AGE":
        age_norm = max(parent1.age, parent2.age) / float(max(1, GA_MAX_AGE))
        prob = base_mu * age_norm                                       # older → higher μ
    else:  # "NONE" or fall‑back
        prob = base_mu

    # clamp to configured bounds
    prob = max(_INDIV_MUT_MIN, min(_INDIV_MUT_MAX, prob))
    return prob
# ------------------------------------------------------------------ #

# ───────────────────── VISUALIZATION (Step B) ───────────────────── #

def plot_dtsp_paths(path_pair, title="DTSP – two disjoint tours"):
    """
    מצייר את שני המסלולים על גרף פיזור.
    כחול – מסלול 1, כתום – מסלול 2.
    """
    if not SHOW_PLOTS:
        return
    if not DTSP_CITIES:
        print("No cities loaded – cannot plot")
        return
    import matplotlib.pyplot as plt
    path1, path2 = path_pair
    xs = [c[0] for c in DTSP_CITIES]
    ys = [c[1] for c in DTSP_CITIES]
    plt.figure(figsize=(6, 6))

    for path, style in [(path1, {"color": "tab:blue"}),
                        (path2, {"color": "tab:orange"})]:
        for i in range(len(path)):
            a, b = path[i], path[(i + 1) % len(path)]
            plt.plot([DTSP_CITIES[a][0], DTSP_CITIES[b][0]],
                     [DTSP_CITIES[a][1], DTSP_CITIES[b][1]],
                     **style, alpha=0.7)

    plt.scatter(xs, ys, c="k", s=30, zorder=5)
    for idx, (x, y) in enumerate(DTSP_CITIES):
        plt.text(x, y, str(idx), fontsize=8, ha="right", va="bottom")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# ───────────────────── 2-OPT LOCAL IMPROVEMENT (Step F) ─────────── #

def two_opt(path):
    """
    Heuristic 2-opt: הופך קטעים שמקצרים את המסלול.
    """
    improved = True
    n = len(path)
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n if i else n - 1):
                a, b = path[i], path[(i + 1) % n]
                c, d = path[j], path[(j + 1) % n]
                if _dist(a, c) + _dist(b, d) < _dist(a, b) + _dist(c, d) - 1e-8:
                    path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                    improved = True
    return path

def lcs_length(s, t):
    """
    Compute the length of the Longest Common Subsequence (LCS) between strings s and t.
    This function is used for the LCS-based fitness heuristic.
    """
    m, n = len(s), len(t)
    # Initialize a matrix dp with dimensions (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if s[i] == t[j]:
                dp[i+1][j+1] = dp[i][j] + 1  # Increase count if characters match
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    return dp[m][n]

class GAIndividual:
    """
    Class representing an individual (candidate solution) in the genetic algorithm.
    The representation depends on the mode: string (for Hello World), ARC puzzles, or bin packing.
    """
    def __init__(self, representation=None):
        # If a representation is provided, use it; otherwise, generate a random one based on the current mode.
        if GA_MODE == "DTSP":
            # ייצוג: שני מסלולים כתמורות (permutations) של הערים
            self.repr = representation if representation else self.random_dtsp()
        else:
            self.repr = representation if representation is not None else self.random_repr()
        self.fitness = 0  # The fitness value of the individual (lower is better for our minimization problems)
        self.age = 0      # The age in generations (used for aging-based survivor selection)
        # cache for behavioural‑distance calculations
        self._edge_union = None
        self._perm_index = None
        if GA_MODE == "BINPACKING":
            self._update_perm_index()

    def random_repr(self):
        """Generate a random representation based on the current GA mode."""
        if GA_MODE == "STRING":
            # Create a random string of the same length as GA_TARGET
            return ''.join(chr(random.randint(32, 122)) for _ in range(len(GA_TARGET)))
        elif GA_MODE == "ARC":
            # Duplicate the input grid and perform a few random mutations
            grid = [row.copy() for row in GA_ARC_INPUT_GRID]
            for _ in range(random.randint(1, 5)):
                self.mutate_grid(grid)
            return grid
        elif GA_MODE == "BINPACKING":
            # Create a random permutation of indices representing the order of items
            n = len(BP_ITEMS)
            perm = list(range(n))
            random.shuffle(perm)
            return perm

    def random_dtsp(self):
        """
        Generate two completely random tours over all cities.
        The two tours are sampled independently, therefore edge
        overlaps between them **are allowed** at this stage.
        """
        # two *independent* random tours
        cities = list(range(len(DTSP_CITIES)))
        path1 = random.sample(cities, len(cities))   # completely random permutation
        path2 = random.sample(cities, len(cities))   # another independent permutation
        return (path1, path2)

    def mutate_grid(self, grid):
        """Randomly mutate the grid by changing one random cell's value."""
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        i = random.randint(0, rows - 1)
        j = random.randint(0, cols - 1)
        grid[i][j] = random.randint(0, 9)

    def calculate_fitness(self):
        """
        Calculate fitness for the individual.
        In STRING mode, it computes the sum of absolute differences per character.
        In ARC mode, it counts the number of mismatching cells.
        """
        if GA_MODE == "STRING":
            self.fitness = sum(abs(ord(self.repr[i]) - ord(GA_TARGET[i])) for i in range(len(GA_TARGET)))
        elif GA_MODE == "ARC":
            match_count = 0
            for i in range(len(GA_ARC_TARGET_GRID)):
                for j in range(len(GA_ARC_TARGET_GRID[0])):
                    if self.repr[i][j] == GA_ARC_TARGET_GRID[i][j]:
                        match_count += 1
            total_cells = len(GA_ARC_TARGET_GRID) * len(GA_ARC_TARGET_GRID[0])
            self.fitness = total_cells - match_count

    # ---------- Task 7 ----------
    def calculate_fitness_lcs(self):
        """
        Calculate fitness based on the Longest Common Subsequence (LCS) method,
        adjusted with an offset and bonus.
        """
        if GA_MODE == "STRING":
            lcs = lcs_length(self.repr, GA_TARGET)
            bonus = sum(1 for i in range(len(GA_TARGET)) if self.repr[i] == GA_TARGET[i])
            offset = GA_BONUS_FACTOR * len(GA_TARGET)
            self.fitness = (len(GA_TARGET) - lcs) - (GA_BONUS_FACTOR * bonus) + offset

    def calculate_fitness_binpacking(self):
        """
        Calculate fitness for bin packing:
        Use a First-Fit approach to place items into bins, ensuring that
        the sum of item sizes in each bin does not exceed BP_CAPACITY.
        The fitness is the number of bins used minus the theoretical optimal.
        """
        bins = []
        for i in self.repr:
            item_size = BP_ITEMS[i]
            placed = False
            for b in bins:
                if sum(BP_ITEMS[j] for j in b) + item_size <= BP_CAPACITY:
                    b.append(i)
                    placed = True
                    break
            if not placed:
                bins.append([i])
        num_bins = len(bins)
        self.fitness = num_bins - BP_OPTIMAL

    def calculate_fitness_dtsp(self):
        path1, path2 = self.repr

        # אורך כל מסלול
        total1 = self._path_distance(path1)
        total2 = self._path_distance(path2)

        # בונה סט קשתות לא מכוּוָנות לכל מסלול
        edges1 = {
            tuple(sorted((path1[i], path1[(i + 1) % len(path1)])))
            for i in range(len(path1))
        }
        edges2 = {
            tuple(sorted((path2[i], path2[(i + 1) % len(path2)])))
            for i in range(len(path2))
        }

        # קשתות משותפות לשני המסלולים (ללא כיוון)
        common_edges = edges1 & edges2
        penalty = len(common_edges) * 10_000  # 10 מיל׳ לכל חפיפה

        # כושר = אורך המסלול הארוך + קנס
        self.fitness = max(total1, total2) + penalty

    def _path_distance(self, path):
        total = 0.0
        n = len(path)
        for i in range(n):
            total += _dist(path[i], path[(i + 1) % n])
        return total

    def _get_edges(self, path):
        """גרסה מתוקנת ללא הסרת קשתות כפולים"""
        edges = []
        n = len(path)
        for i in range(n):
            a = path[i]
            b = path[(i+1) % n]
            edges.append((a, b))
        return edges  # מחזיר את כל הקשתות כולל כפילויות באותו מסלול

    # ───────── cached union‑of‑edges helper (for novelty) ────────
    def _get_edge_union(self):
        """
        Return a *cached* set of undirected edges across **both** DTSP tours.
        Re‑computes the set only if the cache is invalidated (None).
        """
        if self._edge_union is not None or GA_MODE != "DTSP":
            return self._edge_union
        path1, path2 = self.repr
        def edge_set(path):
            return {tuple(sorted((path[i], path[(i + 1) % len(path)])))
                    for i in range(len(path))}
        self._edge_union = edge_set(path1) | edge_set(path2)
        return self._edge_union

    def mutate(self):
        """
        Apply mutation to the individual.
        In STRING mode, change a random character.
        In BINPACKING mode, perform a swap mutation on the permutation.
        """
        if GA_MODE == "STRING":
            pos = random.randint(0, len(self.repr) - 1)
            delta = chr((ord(self.repr[pos]) + random.randint(0, 90)) % 122)
            s = list(self.repr)
            s[pos] = delta
            self.repr = ''.join(s)
        elif GA_MODE == "BINPACKING":
            a, b = random.sample(range(len(self.repr)), 2)
            self.repr[a], self.repr[b] = self.repr[b], self.repr[a]
            # keep cached positions in sync
            if self._perm_index is not None:
                self._perm_index[self.repr[a]], self._perm_index[self.repr[b]] = a, b
            # 2-opt not applicable; ensure _update_perm_index is called only if cache is missing
            if self._perm_index is None:
                self._update_perm_index()
        elif GA_MODE == "DTSP":
            self.mutate_dtsp()
    def _update_perm_index(self):
        """Cache a mapping item->position for fast rank distance (Bin Packing novelty)."""
        if GA_MODE == "BINPACKING":
            self._perm_index = {item: idx for idx, item in enumerate(self.repr)}

    def mutate_dtsp(self):
        """מוטציה משופרת עם היפוך תת-מסלול"""
        path_idx = random.randint(0, 1)
        path = self.repr[path_idx]
        
        # בחירת סוג מוטציה אקראית
        if random.random() < 0.5:
            # החלפת שני ערים
            a, b = random.sample(range(len(path)), 2)
            path[a], path[b] = path[b], path[a]
        else:
            # היפוך תת-מסלול
            start, end = sorted(random.sample(range(len(path)), 2))
            path[start:end+1] = path[start:end+1][::-1]
        self.repair_dtsp()  # הוספת תיקון אוטומטי לאחר מוטציה
        self._edge_union = None

    def repair_dtsp(self):
        """
        Aggressively break *all* edge overlaps between the two tours.
        Repeats up to 50 attempts, each time locating the current set
        of overlapping edges and applying a 2‑opt‑style segment reversal
        to one of the tours until **no** common edges remain or the safety
        limit is reached.
        """
        path1, path2 = self.repr
        max_attempts = 50
        attempts = 0

        while tours_edge_overlap(path1, path2) and attempts < max_attempts:
            edges1 = self._get_edges(path1)
            edges2 = self._get_edges(path2)
            common_edges = list(set(edges1).intersection(edges2))
            if not common_edges:
                break  # overlap already resolved
            # pick one offending edge at random and flip a segment that contains it
            a, b = random.choice(common_edges)

            # helper: find index of edge (x,y) or (y,x) in a path
            def _edge_index(p, x, y):
                try:
                    i = p.index(x)
                    if p[(i + 1) % len(p)] == y:
                        return i
                    i = p.index(y)
                    if p[(i + 1) % len(p)] == x:
                        return i
                except ValueError:
                    return None
                return None

            # decide which tour to modify this round
            if random.random() < 0.5:
                idx = _edge_index(path1, a, b)
                if idx is not None:
                    i, j = sorted((idx, (idx + 1) % len(path1)))
                    path1[i:j + 1] = reversed(path1[i:j + 1])
            else:
                idx = _edge_index(path2, a, b)
                if idx is not None:
                    i, j = sorted((idx, (idx + 1) % len(path2)))
                    path2[i:j + 1] = reversed(path2[i:j + 1])

            attempts += 1

        # invalidate cached union so novelty / fitness reflect the update
        self._edge_union = None
    
    def local_optimize_dtsp(self):
        """
        Apply 2‑opt independently to both tours (Step F) **and immediately
        repair any edge overlaps** that this local optimisation might introduce.
        This prevents the periodic fitness spikes observed every 50 generations.
        """
        p1, p2 = self.repr

        # --- run independent 2‑opt on each tour -----------------
        two_opt(p1)
        two_opt(p2)

        # --- iterative repair to guarantee *full* edge disjointness ----
        # The simple one‑shot repair sometimes leaves residual overlaps
        # and causes the enormous fitness spikes seen every 50 generations.
        # We therefore loop until no overlap remains (or a small cap
        # is reached to avoid infinite loops on pathological cases).
        max_attempts = 30
        attempts = 0
        while tours_edge_overlap(p1, p2) and attempts < max_attempts:
            self.repair_dtsp()
            attempts += 1

        # invalidate cached union so novelty / fitness reflect the update
        self._edge_union = None

def init_population():
    """Initialize and return a list of GAIndividual objects forming the initial population."""
    return [GAIndividual() for _ in range(GA_POPSIZE)]

def sort_population(population):
    """Sort the population in increasing order of fitness (lower is better)."""
    population.sort(key=lambda ind: ind.fitness)

def elitism(population, buffer, esize):
    """
    Preserve the top esize individuals into the new population buffer.
    Creates a copy of those individuals to ensure they are not altered by future mutations.
    """
    buffer[:esize] = [
        GAIndividual(copy.deepcopy(ind.repr))  # שימוש בעותק עמוק לייצוגים מורכבים
        for ind in population[:esize]
    ]
    for i in range(esize):
        buffer[i].fitness = population[i].fitness
        buffer[i].age = population[i].age

# ---------- Task 4: Crossover Operators ----------
def crossover_single(parent1, parent2):
    """Perform single-point crossover between two parents."""
    tsize = len(parent1.repr)
    spos = random.randint(0, tsize - 1)
    return parent1.repr[:spos] + parent2.repr[spos:]

def crossover_two(parent1, parent2):
    """Perform two-point crossover between two parents."""
    tsize = len(parent1.repr)
    if tsize < 2:
        return crossover_single(parent1, parent2)
    point1 = random.randint(0, tsize - 2)
    point2 = random.randint(point1 + 1, tsize - 1)
    return parent1.repr[:point1] + parent2.repr[point1:point2] + parent1.repr[point2:]

def crossover_uniform(parent1, parent2):
    """Perform uniform crossover between two parents."""
    tsize = len(parent1.repr)
    child_chars = []
    for i in range(tsize):
        child_chars.append(parent1.repr[i] if random.random() < 0.5 else parent2.repr[i])
    return ''.join(child_chars)

def crossover_trivial(parent1, parent2):
    """Return one of the parents at random."""
    return parent1.repr if random.random() < 0.5 else parent2.repr

def crossover_grid(parent1, parent2):
    """Crossover for ARC mode using a grid split."""
    grid1 = parent1.repr
    grid2 = parent2.repr
    child_grid = [row.copy() for row in grid1]
    if random.random() < 0.5:
        split_row = random.randint(1, len(grid1) - 1)
        child_grid[split_row:] = grid2[split_row:]
    else:
        split_col = random.randint(1, len(grid1[0]) - 1)
        for i in range(len(grid1)):
            child_grid[i][split_col:] = grid2[i][split_col:]
    return child_grid

def ox_crossover(p1, p2):
    """אופרטור OX כללי לעבודה עם רשימות ישירות"""
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b] = p1[a:b]
    pos = b
    for gene in p2[b:] + p2[:b]:
        if gene not in child:
            if pos >= size:
                pos = 0
            child[pos] = gene
            pos += 1
    return child

def crossover_binpacking(parent1, parent2):
    """אופרטור OX עבור Bin Packing"""
    return ox_crossover(parent1.repr, parent2.repr)

def crossover_dtsp(parent1, parent2):
    """אופרטור crossover עבור DTSP"""
    child_path1 = ox_crossover(parent1.repr[0], parent2.repr[0])
    child_path2 = ox_crossover(parent1.repr[1], parent2.repr[1])
    return (child_path1, child_path2)

# ---------- Task 10: Parent Selection Methods ----------
def select_parent_RWS(population):
    """Select a parent using Roulette Wheel Selection (fitness-proportional).
    Minimisation is assumed – therefore the selection probability is
    proportional to (worst − fitness)."""
    worst = max(ind.fitness for ind in population)
    adjusted = [worst - ind.fitness for ind in population]
    total = sum(adjusted)
    if total == 0:
        return random.choice(population)
    r = random.uniform(0, total)
    cum = 0.0
    for ind, val in zip(population, adjusted):
        cum += val
        if cum >= r:
            return ind
    return population[-1]

def select_parent_TournamentDet(population):
    """Select a parent using deterministic tournament selection."""
    candidates = random.sample(population, GA_TOURNAMENT_K)
    return min(candidates, key=lambda ind: ind.fitness)

def select_parent_TournamentStoch(population):
    """Select a parent using stochastic tournament selection."""
    candidates = random.sample(population, GA_TOURNAMENT_K)
    candidates.sort(key=lambda ind: ind.fitness)
    for candidate in candidates:
        if random.random() < GA_TOURNAMENT_P:
            return candidate
    return candidates[-1]

def select_parents_SUS(population, num_parents):
    """
    Select parents using Stochastic Universal Sampling (SUS).
    This method provides a more even chance of selection.
    """
    worst = max(ind.fitness for ind in population)
    adjusted = [worst - ind.fitness for ind in population]
    total = sum(adjusted)
    if total == 0:
        return [random.choice(population) for _ in range(num_parents)]
    step = total / num_parents
    start = random.uniform(0, step)
    pointers = [start + i * step for i in range(num_parents)]
    parents = []
    for p in pointers:
        cum = 0
        for ind, val in zip(population, adjusted):
            cum += val
            if cum >= p:
                parents.append(ind)
                break
    return parents

def select_parent_Original(population):
    """Select a random parent from the top half of the population."""
    return random.choice(population[:len(population)//2])

# ---------- Task 10: Aging Survivor Selection ----------
def apply_aging(population):
    """
    Increment the age of each individual.
    Remove individuals that exceed GA_MAX_AGE and replace them with new random individuals.
    """
    survivors = []
    for ind in population:
        ind.age += 1
        if ind.age < GA_MAX_AGE:
            survivors.append(ind)
    while len(survivors) < GA_POPSIZE:
        new_ind = GAIndividual()
        new_ind.age = 0
        survivors.append(new_ind)
    return survivors

# ---------- Task 9: Genetic Diversity Metrics (Factor Exploration) ----------
def compute_diversity_metrics(population):
    """
    Compute diversity metrics: 
    Average pairwise Hamming distance, average distinct alleles per gene,
    and average Shannon entropy per gene.
    """
    if GA_MODE == "STRING":
        L = len(GA_TARGET)
        N = len(population)
        total_hamming = 0.0
        total_distinct = 0
        total_entropy = 0.0
        for j in range(L):
            freq = {}
            for ind in population:
                allele = ind.repr[j]
                freq[allele] = freq.get(allele, 0) + 1
            pos_p2_sum = sum((count / N) ** 2 for count in freq.values())
            pos_entropy = -sum((count / N) * math.log2(count / N) for count in freq.values() if count > 0)
            avg_diff = 1 - pos_p2_sum
            total_hamming += avg_diff
            total_distinct += len(freq)
            total_entropy += pos_entropy
        avg_hamming_distance = total_hamming
        avg_distinct = total_distinct / L
        avg_entropy = total_entropy / L
        return avg_hamming_distance, avg_distinct, avg_entropy
    elif GA_MODE == "ARC":
        rows = len(GA_ARC_TARGET_GRID)
        cols = len(GA_ARC_TARGET_GRID[0])
        total_distinct = 0
        total_entropy = 0.0
        for i in range(rows):
            for j in range(cols):
                freq = {}
                for ind in population:
                    val = ind.repr[i][j]
                    freq[val] = freq.get(val, 0) + 1
                total_distinct += len(freq)
                for count in freq.values():
                    p = count / GA_POPSIZE
                    total_entropy -= p * math.log2(p) if p > 0 else 0
        avg_distinct = total_distinct / (rows * cols)
        avg_entropy = total_entropy / (rows * cols)
        return 0, avg_distinct, avg_entropy
    elif GA_MODE == "DTSP":
        # מדדי גיוון עבור מסלולים
        unique_paths = len({tuple(ind.repr[0] + ind.repr[1]) for ind in population})
        return (0, unique_paths, 0)
    else:  # Bin Packing
        return (0, 0, 0)


# ────────── Helper for Task 2‑c: novelty score ────────── #
def _behavioural_distance(ind1: 'GAIndividual', ind2: 'GAIndividual') -> float:
    """
    A quick, domain‑agnostic distance between two individuals.
    • STRING   : normalised Hamming distance.
    • BINPACK  : Kendall‑Tau distance between permutations (normalised).
    • DTSP     : fraction of *differing* undirected edges across *both* tours.
    • ARC      : Hamming distance of flattened grids (normalised).
    The distance is mapped to [0,1].
    """
    if GA_MODE == "STRING":
        L = len(GA_TARGET)
        return sum(a != b for a, b in zip(ind1.repr, ind2.repr)) / L
    elif GA_MODE == "BINPACKING":
        # Spearman Footrule distance (O(n)) using cached indices
        perm1, perm2 = ind1.repr, ind2.repr
        n = len(perm1)
        # ensure caches are ready
        if ind1._perm_index is None:
            ind1._update_perm_index()
        if ind2._perm_index is None:
            ind2._update_perm_index()
        diff = sum(abs(ind1._perm_index[item] - ind2._perm_index[item]) for item in perm1)
        max_diff = n * (n - 1) / 2  # maximal possible footrule distance
        return diff / max_diff
    elif GA_MODE == "DTSP":
        e1 = ind1._get_edge_union()
        e2 = ind2._get_edge_union()
        if not e1 or not e2:
            return 0.0
        diff  = len(e1.symmetric_difference(e2))
        union = len(e1 | e2)
        return diff / union if union else 0.0
    elif GA_MODE == "ARC":
        flat1 = sum(ind1.repr, [])
        flat2 = sum(ind2.repr, [])
        L = len(flat1)
        return sum(a != b for a, b in zip(flat1, flat2)) / L
    return 0.0

def compute_novelty(ind: 'GAIndividual', population: list['GAIndividual']) -> float:
    """
    Behavioural novelty ≈ average distance to k‑nearest neighbours.
    Uses a *random sample* of at most NOVELTY_SAMPLE peers to keep the
    complexity roughly O(k·log k) instead of O(N²).
    """
    if len(population) <= 1:
        return 0.0
    sample = random.sample(
        [peer for peer in population if peer is not ind],
        k=min(NOVELTY_SAMPLE, len(population) - 1)
    )
    dists = sorted(_behavioural_distance(ind, peer) for peer in sample)
    k = min(NOVELTY_K, len(dists))
    return sum(dists[:k]) / k if k else 0.0


# ───────── Niching / Speciation helper (Task 4) ───────── #
def _apply_niching(population: list['GAIndividual']):
    """
    Post‑process `.fitness` according to GA_NICHING_ALGO **after**
    raw/auxiliary fitnesses were set.  Works with the generic
    _behavioural_distance() already available for all modes.

    • "SHARING"    : multiply fitness by (1 + Σ sh(d_ij)) where
                     sh(d) = 1 – d/r  for d < r,   else 0
                     (minimisation ⇒ worse when too crowded).

    • "SPECIATION" : cluster individuals greedily; every individual is
                     assigned to the first representative closer than
                     SPECIES_THRESHOLD.  Fitness is penalised by
                     SPECIES_PENALTY·( |species| – 1 ).
    """
    if GA_NICHING_ALGO == "NONE":
        return

    if GA_NICHING_ALGO == "SHARING":
        n = len(population)
        for i, ind_i in enumerate(population):
            share_sum = 0.0
            for j, ind_j in enumerate(population):
                if i == j:
                    continue
                d = _behavioural_distance(ind_i, ind_j)
                if d < SHARING_RADIUS:
                    share_sum += 1.0 - (d / SHARING_RADIUS)
            if share_sum > 0.0:
                ind_i.fitness *= (1.0 + share_sum)   # minimisation ⇒ up‑scale

    elif GA_NICHING_ALGO == "SPECIATION":
        representatives: list['GAIndividual'] = []
        species: list[list['GAIndividual']] = []
        # --- greedy assignment -----------------------------------------
        for ind in population:
            assigned = False
            for s_idx, rep in enumerate(representatives):
                if _behavioural_distance(ind, rep) < SPECIES_THRESHOLD:
                    species[s_idx].append(ind)
                    assigned = True
                    break
            if not assigned:
                representatives.append(ind)
                species.append([ind])

        # penalty grows with species size
        for sp in species:
            if len(sp) <= 1:
                continue
            penalty = SPECIES_PENALTY * (len(sp) - 1)
            for ind in sp:
                ind.fitness += penalty

# ───── Routine to augment fitness with g(x,t) (Task 2‑c) ───── #
def apply_adaptive_fitness(population: list['GAIndividual']):
    """
    Modify each individual's `.fitness` in‑place according to GA_ADAPT_FIT_POLICY.
    This happens *after* raw fitness was computed and *before* selection/sort.
    The transformation preserves ordering for minimisation.
    Also logs ⟨g(x,t)⟩ for analysis/plots.
    (Optimised distance‑matrix shortcut when GA_ADAPT_FIT_POLICY == 'NOVELTY')
    """
    global CURRENT_GENERATION
    # Skip expensive novelty calculation except every NOVELTY_PERIOD generations
    if GA_ADAPT_FIT_POLICY == "NOVELTY" and CURRENT_GENERATION % NOVELTY_PERIOD != 0:
        # keep last logged g(x,t) value so the plot length stays in sync
        GEN_AVG_G_HISTORY.append(GEN_AVG_G_HISTORY[-1] if GEN_AVG_G_HISTORY else 0.0)
        # --- Task 4: niching/speciation post‑processing ---
        _apply_niching(population)
        return
    if GA_ADAPT_FIT_POLICY == "NONE":
        GEN_AVG_G_HISTORY.append(0.0)
        # --- Task 4: niching/speciation post‑processing ---
        _apply_niching(population)
        return
    g_vals = []
    if GA_ADAPT_FIT_POLICY == "AGE":
        for ind in population:
            g = AGE_WEIGHT * ind.age
            ind.fitness += g                # older ⇒ worse (minimisation)
            g_vals.append(g)
    elif GA_ADAPT_FIT_POLICY == "NOVELTY":
        # --- Optimised novelty computation ---
        # Pre‑compute a symmetric distance matrix once per generation to
        # avoid O(N²) repeated calls to _behavioural_distance inside
        # compute_novelty().  This keeps runtime manageable even for large
        # populations.
        n = len(population)
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = _behavioural_distance(population[i], population[j])
                dist[i][j] = d
                dist[j][i] = d

        k = min(NOVELTY_K, n - 1)  # k‑NN neighbourhood
        novs = []
        for i in range(n):
            # exclude self‑distance (0) and take k nearest neighbours
            neighbours = sorted(dist[i][:i] + dist[i][i + 1:])
            nov = sum(neighbours[:k]) / k if k else 0.0
            novs.append(nov)

        # Apply novelty reward (encourages exploration)
        for ind, nov in zip(population, novs):
            g = -NOVELTY_WEIGHT * nov    # more novel ⇒ *lower* fitness (minimisation)
            ind.fitness += g
            g_vals.append(nov)
    GEN_AVG_G_HISTORY.append(sum(g_vals) / len(g_vals) if g_vals else 0.0)
    # --- Task 4: niching/speciation post‑processing ---
    _apply_niching(population)

# ---------- Task 1: Generation Stats, Task 8 & Task 9 Combined ----------
def print_generation_stats(population, generation, tick_duration, total_elapsed, best_ever_fitness=None):
    """
    Print detailed stats for the current generation, including fitness distribution and selection metrics.
    """
    fitness_values = [ind.fitness for ind in population]
    best = population[0]
    worst = population[-1]
    avg_fitness = sum(fitness_values) / len(fitness_values)
    std_dev = statistics.stdev(fitness_values)
    fitness_range = worst.fitness - best.fitness
    if GA_MODE == "STRING":
        best_repr = f"'{best.repr}'"
    else:
        best_repr = f"{best.repr}"
    print(f"Gen {generation}: Best = {best_repr} (Fitness = {best.fitness})")
    print(f"  Avg Fitness = {avg_fitness:.2f}")
    print(f"  Std Dev = {std_dev:.2f}")
    print(f"  Worst Fitness = {worst.fitness}")
    print(f"  Fitness Range = {fitness_range}")
    if best_ever_fitness is not None:
        print(f"  Best‑Ever Fitness      = {best_ever_fitness}")
    print(f"  Tick Duration (sec) = {tick_duration:.4f}")
    print(f"  Total Elapsed Time (sec) = {total_elapsed:.4f}")
    adjusted = [worst.fitness - ind.fitness for ind in population]
    mean_adjusted = sum(adjusted) / len(adjusted)
    std_adjusted = statistics.stdev(adjusted)
    selection_variance = std_adjusted / mean_adjusted if mean_adjusted != 0 else 0
    total_adjusted = sum(adjusted)
    if total_adjusted == 0:
        probabilities = [1.0 / len(population)] * len(population)
    else:
        probabilities = [val / total_adjusted for val in adjusted]
    top_k = max(1, int(0.1 * len(population)))
    top_avg = sum(probabilities[:top_k]) / top_k
    overall_avg = 1.0 / len(population)
    top_avg_ratio = top_avg / overall_avg 
    print(f"  Selection Variance = {selection_variance:.6f}")
    print(f"  Top-Average Selection Probability Ratio = {top_avg_ratio:.2f}")
    avg_hamming_distance, avg_distinct, avg_entropy = compute_diversity_metrics(population)
    if GA_MODE == "DTSP":
        print(f"  Unique Path Combinations = {avg_distinct}")
    else:
        print(f"  Avg Pairwise Hamming Distance = {avg_hamming_distance:.2f}")
        print(f"  Avg Number of Distinct Alleles per Gene = {avg_distinct:.2f}")
        print(f"  Avg Shannon Entropy per Gene (bits) = {avg_entropy:.2f}")
    print()

# ---------- Task 10: Mating Function ----------
def mate(population, buffer, mutlog=None):
    """
    Create offspring using the selected crossover and mutation operators.
    Elitism is applied to preserve the top individuals.
    """
    global GA_DYNAMIC_MUTRATE
    esize = int(GA_POPSIZE * GA_ELITRATE)
    mut_probs = []          # collect μ_i of this generation
    # average fitness for FITNESS‑based individual µ
    avg_population_fitness = sum(ind.fitness for ind in population) / len(population)
    elitism(population, buffer, esize)
    num_offspring = GA_POPSIZE - esize
    sus_parents = []
    if GA_PARENT_SELECTION_METHOD == "SUS":
        sus_parents = select_parents_SUS(population, num_offspring * 2)
    for i in range(esize, GA_POPSIZE):
        if GA_PARENT_SELECTION_METHOD == "RWS":
            parent1 = select_parent_RWS(population)
            parent2 = select_parent_RWS(population)
        elif GA_PARENT_SELECTION_METHOD == "TournamentDet":
            parent1 = select_parent_TournamentDet(population)
            parent2 = select_parent_TournamentDet(population)
        elif GA_PARENT_SELECTION_METHOD == "TournamentStoch":
            parent1 = select_parent_TournamentStoch(population)
            parent2 = select_parent_TournamentStoch(population)
        elif GA_PARENT_SELECTION_METHOD == "SUS":
            parent1 = sus_parents.pop(0)
            parent2 = sus_parents.pop(0)
        elif GA_PARENT_SELECTION_METHOD == "Original":
            parent1 = select_parent_Original(population)
            parent2 = select_parent_Original(population)
        else:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
        if GA_MODE == "STRING":
            if GA_CROSSOVER_OPERATOR == "SINGLE":
                child_repr = crossover_single(parent1, parent2)
            elif GA_CROSSOVER_OPERATOR == "TWO":
                child_repr = crossover_two(parent1, parent2)
            elif GA_CROSSOVER_OPERATOR == "UNIFORM":
                child_repr = crossover_uniform(parent1, parent2)
            elif GA_CROSSOVER_OPERATOR == "TRIVIAL":
                child_repr = crossover_trivial(parent1, parent2)
        elif GA_MODE == "ARC":
            child_repr = crossover_grid(parent1, parent2)
        elif GA_MODE == "BINPACKING":
            child_repr = crossover_binpacking(parent1, parent2)
        elif GA_MODE == "DTSP":
            child_repr = crossover_dtsp(parent1, parent2)
        child = GAIndividual(child_repr)
        mut_prob = compute_individual_mutrate(parent1, parent2, avg_population_fitness)
        mut_probs.append(mut_prob)
        if random.random() < mut_prob:
            child.mutate()
        buffer.append(child)
    if mutlog is not None and mut_probs:
        mutlog.append(sum(mut_probs) / len(mut_probs))

def plot_grids(input_grid, target_grid, solution_grid=None):
    """
    Display three side-by-side plots for ARC mode:
    - The input grid
    - The target grid
    - The solution grid
    """
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131)
    ax1.matshow(input_grid, cmap='viridis')
    ax1.set_title("Input Grid")
    for (i, j), val in np.ndenumerate(input_grid):
        ax1.text(j, i, f'{val}', ha='center', va='center', color='w' if val > 5 else 'k')
    ax2 = fig.add_subplot(132)
    ax2.matshow(target_grid, cmap='viridis')
    ax2.set_title("Target Grid")
    for (i, j), val in np.ndenumerate(target_grid):
        ax2.text(j, i, f'{val}', ha='center', va='center', color='w' if val > 5 else 'k')
    ax3 = fig.add_subplot(133)
    if solution_grid is not None:
        ax3.matshow(solution_grid, cmap='viridis')
        ax3.set_title("Solution Grid")
        for (i, j), val in np.ndenumerate(solution_grid):
            ax3.text(j, i, f'{val}', ha='center', va='center', color='w' if val > 5 else 'k')
    else:
        ax3.axis('off')
        ax3.text(0.5, 0.5, 'No solution found', ha='center', va='center')
    plt.tight_layout()
    plt.show()

def print_bin_details(bins, total_runtime):
    """
    Print the final bin packing result in the requested format.
    Each bin is numbered sequentially starting from 1, and displays the list of item sizes 
    and the total sum in that bin.
    """
    deviation = len(bins) - BP_OPTIMAL
    print(f"Results for {bp_instance['name']}:")
    print(f"Bins used: {len(bins)}")
    print(f"Theoretical minimum: {BP_OPTIMAL}")
    print(f"Deviation from optimal: {deviation} bins")
    print(f"Runtime: {total_runtime:.4f}s")
    print("Bin details:")
    for idx, b in enumerate(bins, start=1):
        sizes = [BP_ITEMS[i] for i in b]
        total = sum(sizes)
        print(f"Bin {idx}: {sizes}  (Total: {total}/{BP_CAPACITY})")

def load_dtsp_data(cities_path: str):
    """
    Load coordinates from either a CSV file (id,x,y,…) or a TSPLIB *.tsp file.
    Populates DTSP_CITIES and clears the distance cache.
    """
    global DTSP_CITIES, _DISTANCE_CACHE
    DTSP_CITIES = []
    _DISTANCE_CACHE.clear()

    ext = Path(cities_path).suffix.lower()
    if ext == ".csv":
        import csv
        with open(cities_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip optional header
            for row in reader:
                try:
                    DTSP_CITIES.append((float(row[1]), float(row[2])))
                except (IndexError, ValueError):
                    continue
    else:  # assume TSPLIB format
        with open(cities_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        try:
            start = lines.index("NODE_COORD_SECTION") + 1
        except ValueError:
            raise ValueError("TSPLIB file missing NODE_COORD_SECTION")
        for ln in lines[start:]:
            if ln.upper().startswith("EOF"):
                break
            parts = ln.split()
            if len(parts) >= 3:
                _, x, y = parts[:3]
                DTSP_CITIES.append((float(x), float(y)))

    if len(DTSP_CITIES) < 3:
        raise ValueError(f"Failed to load coordinates from {cities_path}")
    print(f"Loaded {len(DTSP_CITIES)} cities from {Path(cities_path).name}")


# ───────────────────── DTSP FULL 10‑STEP PIPELINE ───────────────────── #

def run_dtsp(cities_path: str):
    """
    End‑to‑end pipeline implementing the 10 steps (A–J) for solving the
    Disjoint‑TSP (DTSP).  Each major stage is marked so that the execution
    trace is easy to follow and debug.

        A) Load coordinates                         – already performed upstream
        B) Visualise random initial tours           (optional)
        C) Initialise GA population
        D) Evaluate fitness of initial population
        E) Evolutionary loop with logging
        F) Periodic 2‑opt local optimisation
        G) Stagnation detection + triggered hyper‑mutation
        H) Plot convergence curves
        I) Visualise best pair of tours
        J) Persist best solution to <cities>.best.json
    """
    global CURRENT_GENERATION
    # --- user‑tunable patience: stop after this many generations w/o progress
    EARLY_STOP_PATIENCE = 200        # ← was effectively 300 before
    global GA_MODE, GA_CROSSOVER_OPERATOR, GA_FITNESS_HEURISTIC

    # ---------- Step B: draw a random initial pair of tours -------------
    if len(DTSP_CITIES) >= 3:
        sample_ind = GAIndividual()
        plot_dtsp_paths(sample_ind.repr, title="Step B – random initial tours")

    # ---------- Step C: GA initialisation -------------------------------
    GA_CROSSOVER_OPERATOR = "DTSP"
    GA_FITNESS_HEURISTIC  = "DTSP"
    population = init_population()
    GA_DYNAMIC_MUTRATE = GA_MUTATIONRATE  # reset effective µ
    buffer     = []

    best_ever_fitness    = float("inf")
    best_ever_individual = None
    no_improve_count     = 0
    fitness_history      = []
    avg_history          = []
    worst_history        = []
    distribution_log     = []
    mutation_rate_history = []
    GEN_AVG_INDIV_MUT_HISTORY.clear()

    start_time = timeit.default_timer()

    # ---------- Step D: initial evaluation ------------------------------
    for ind in population:
        ind.calculate_fitness_dtsp()
    CURRENT_GENERATION = 0   # generation counter initialisation
    apply_adaptive_fitness(population)
    sort_population(population)

    # ---------- Step E + F + G: evolutionary loop -----------------------
    for generation in range(GA_MAXITER):
        CURRENT_GENERATION = generation
        # Step F – local 2‑opt every 25 generations
        if generation and generation % 25 == 0:
            for ind in population:
                ind.local_optimize_dtsp()

        # Re‑evaluate & sort
        for ind in population:
            ind.calculate_fitness_dtsp()
        apply_adaptive_fitness(population)      # ← NEW
        sort_population(population)

        best   = population[0]
        worst  = population[-1]
        avg    = sum(ind.fitness for ind in population) / len(population)
        fitness_history.append(best.fitness)
        avg_history.append(avg)
        worst_history.append(worst.fitness)
        distribution_log.append([ind.fitness for ind in population])

        # Step G – global‑best tracking & stagnation counter
        if best.fitness < best_ever_fitness - 1e-9:
            best_ever_fitness    = best.fitness
            best_ever_individual = copy.deepcopy(best)
            no_improve_count     = 0
        else:
            no_improve_count += 1

        print(f"Gen {generation:5d} | best {best_ever_fitness:,.2f} | "
              f"avg {avg:,.2f} | worst {worst.fitness:,.2f} | "
              f'Δ={no_improve_count}')
        print(f"           µ_pop(t) = {GA_DYNAMIC_MUTRATE:.3f}  |  μ‑policy: {GA_INDIV_MUT_POLICY}")
        if GEN_AVG_INDIV_MUT_HISTORY:
            print(f"           µ_indiv_avg(t) = {GEN_AVG_INDIV_MUT_HISTORY[-1]:.3f}")
        if GEN_AVG_G_HISTORY:
            print(f"           ⟨g(x,t)⟩ = {GEN_AVG_G_HISTORY[-1]:.3f}  |  g‑policy: {GA_ADAPT_FIT_POLICY}")
        mutation_rate_history.append(GA_DYNAMIC_MUTRATE)

        # --- update mutation probability for this generation ---
        GA_DYNAMIC_MUTRATE = compute_mutation_rate(generation, no_improve_count)


        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"⏹  Early‑stop: no improvement for {EARLY_STOP_PATIENCE} generations")
            break

        # Mate & create next generation
        buffer.clear()
        mate(population, buffer, GEN_AVG_INDIV_MUT_HISTORY)
        population, buffer = buffer, population
        population = apply_aging(population)

    # ---------- Step H: convergence plots -------------------------------
    if SHOW_PLOTS:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(fitness_history, label="best")
            plt.plot(avg_history, label="avg")
            plt.plot(worst_history, label="worst")
            plt.xlabel("generation"); plt.ylabel("fitness")
            plt.title("DTSP convergence"); plt.legend(); plt.grid(True)
            plt.tight_layout(); plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")

        # ---------- µ(t) plot -------------------------------------------------
        if mutation_rate_history:
            plt.figure(figsize=(10, 4))
            plt.plot(mutation_rate_history, label="µ(t)")
            plt.xlabel("generation")
            plt.ylabel("mutation rate")
            plt.title("Mutation rate per generation")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        if GEN_AVG_G_HISTORY:
            plt.figure(figsize=(10,4))
            plt.plot(GEN_AVG_G_HISTORY, label="⟨g(x,t)⟩")
            plt.xlabel("generation"); plt.ylabel("auxiliary reward g(x,t)")
            plt.title(f"{GA_ADAPT_FIT_POLICY} reward per generation")
            plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
        if mutation_rate_history and GEN_AVG_INDIV_MUT_HISTORY:
            plt.figure(figsize=(10, 4))
            plt.plot(mutation_rate_history, label="population µ_pop(t)")
            plt.plot(GEN_AVG_INDIV_MUT_HISTORY, label="avg individual µ_indiv(t)")
            plt.xlabel("generation")
            plt.ylabel("mutation rate")
            plt.title("Population vs Individual mutation rate")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    # ---------- Step I: visualise best paths ----------------------------
    # ===== Ensure best solution is disjoint; attempt final repair =====
    if best_ever_individual is not None:
        # --- attempt automatic repair if edge overlap remains ---
        attempts = 0
        while tours_edge_overlap(*best_ever_individual.repr) and attempts < 10:
            best_ever_individual.repair_dtsp()
            best_ever_individual.local_optimize_dtsp()
            best_ever_individual.calculate_fitness_dtsp()
            attempts += 1
        # update stored fitness after possible repair
        best_ever_fitness = best_ever_individual.fitness

        # ===== Print best tours & their lengths for easy inspection =====
        def _path_len(path):
            return sum(_dist(path[i], path[(i + 1) % len(path)])
                       for i in range(len(path)))

        tour1, tour2 = best_ever_individual.repr
        valid1 = is_hamiltonian_cycle(tour1)
        valid2 = is_hamiltonian_cycle(tour2)
        overlap = tours_edge_overlap(tour1, tour2)

        print("\n──────────  BEST DTSP SOLUTION  ──────────")
        print(f"Fitness (max tour length): {best_ever_fitness:,.2f}")
        print(f"Tour 1 length: {_path_len(tour1):,.2f}  |  Valid cycle: {valid1}")
        print(f"Tour 1 order (closed): {tour1 + [tour1[0]]}")
        print(f"Tour 2 length: {_path_len(tour2):,.2f}  |  Valid cycle: {valid2}")
        print(f"Tour 2 order (closed): {tour2 + [tour2[0]]}")
        print(f"Edge overlap between tours: {overlap}")
        print("──────────────────────────────────────────\n")

        if not (valid1 and valid2) or overlap:
            print("⚠️  Warning: final solution violates DTSP constraints "
                  "(non‑Hamiltonian cycle or edge overlap). "
                  "Consider increasing evolution time or enabling repair heuristics.")
        plot_dtsp_paths(best_ever_individual.repr,
                        title=f"Step I – best fitness {best_ever_fitness:,.2f}")

    # ---------- Step J: persist solution --------------------------------
    out_path = Path(cities_path).with_suffix(".best.json")
    try:
        with open(out_path, "w") as f:
            json.dump({"fitness": best_ever_fitness,
                       "path1": best_ever_individual.repr[0],
                       "path2": best_ever_individual.repr[1]}, f, indent=2)
        print(f"💾  Saved best solution to {out_path.name}")
    except Exception as e:
        print(f"Could not save best solution: {e}")

def run_baldwin_experiment():
    import matplotlib.pyplot as plt

    # ---------------- Experimental constants ------------------
    GENOME_LENGTH      = 20
    POP_SIZE           = 1000
    LEARNING_TRIALS    = 1000
    MAX_GENERATIONS    = 200

    # hidden target (fixed for the entire run)
    TARGET = [random.choice('01') for _ in range(GENOME_LENGTH)]

    # helper: create a single random genome according to the required distr.
    def _random_genome():
        genome = []
        for bit in TARGET:
            r = random.random()
            if r < 0.5:
                genome.append('?')
            elif r < 0.75:
                genome.append(bit)              # correct
            else:
                genome.append('1' if bit == '0' else '0')  # incorrect
        return genome

    population = [_random_genome() for _ in range(POP_SIZE)]

    # --- statistics logging ---
    perc_correct = []
    perc_incorrect = []
    perc_question = []

    for gen in range(MAX_GENERATIONS):
        # -------- evaluate fitness with learning -------------
        fitness = []
        for genome in population:
            # quick check: any incorrect hard-wired allele → fitness 1 (n=0)
            impossible = any((g != '?' and g != t) for g, t in zip(genome, TARGET))
            if impossible:
                fitness.append(1.0)  # n = 0 ⇒ f = 1
                continue
            # otherwise simulate up-to LEARNING_TRIALS attempts
            n = 0
            success = False
            for trial in range(LEARNING_TRIALS):
                # fill '?' with random bits
                candidate = [g if g != '?' else random.choice('01') for g in genome]
                if candidate == TARGET:
                    success = True
                    n = LEARNING_TRIALS - trial  # trials left *including* this one
                    break
            if not success:
                n = 0
            fitness.append(1.0 + 19.0 * n / LEARNING_TRIALS)

        # --------- log population averages for the plot ----------
        cnt_correct = 0
        cnt_incorrect = 0
        cnt_q = 0
        for genome in population:
            for a, t in zip(genome, TARGET):
                if a == '?':
                    cnt_q += 1
                elif a == t:
                    cnt_correct += 1
                else:
                    cnt_incorrect += 1
        total_alleles = POP_SIZE * GENOME_LENGTH
        perc_correct.append(100.0 * cnt_correct / total_alleles)
        perc_incorrect.append(100.0 * cnt_incorrect / total_alleles)
        perc_question.append(100.0 * cnt_q / total_alleles)

        # ---------------- reproduction (RWS) ----------------------
        # --- Roulette-Wheel Selection (maximisation) ---
        total_fit = sum(fitness)
        if total_fit == 0:
            probs = [1.0 / POP_SIZE] * POP_SIZE
        else:
            probs = [f / total_fit for f in fitness]

        # cumulative prob table
        cumprob = []
        s = 0.0
        for p in probs:
            s += p
            cumprob.append(s)

        def _select_parent():
            r = random.random()
            for idx, cp in enumerate(cumprob):
                if r <= cp:
                    return population[idx]
            return population[-1]

        new_population = []
        for _ in range(POP_SIZE):
            p1 = _select_parent()
            p2 = _select_parent()
            child = [random.choice([a, b]) for a, b in zip(p1, p2)]
            new_population.append(child)
        population = new_population

    # -------------------- plotting ---------------------------
    gens = list(range(MAX_GENERATIONS))
    plt.figure(figsize=(10, 5))
    plt.plot(gens, perc_correct, label="Correct hard-wired")
    plt.plot(gens, perc_incorrect, label="Incorrect hard-wired")
    plt.plot(gens, perc_question, label="Learnable '?'", linestyle='--')
    plt.xlabel("Generation")
    plt.ylabel("Percentage of loci (%)")
    plt.title("Baldwin Effect – population averages over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------- simple console verdict -----------------
    print("\nFinal averages: ")
    print(f"  Correct   : {perc_correct[-1]:.1f}%")
    print(f"  Incorrect : {perc_incorrect[-1]:.1f}%")
    print(f"  '?' loci  : {perc_question[-1]:.1f}%")
    if perc_incorrect[-1] < 1 and perc_question[-1] < 5:
        print("\nObservation: Incorrect alleles vanished and most '?' loci became genetically fixed – consistent with the Baldwin Effect.")
    else:
        print("\nObservation: The expected pattern (drop in incorrect + '?' loci) is weak — perhaps more generations are needed.")

def main():
    global GA_MODE, GA_ARC_TARGET_GRID, GA_ARC_INPUT_GRID, bp_instance
    global GA_FITNESS_HEURISTIC, GA_CROSSOVER_OPERATOR
    global CURRENT_GENERATION

    print("Select mode:")
    print("1 - String evolution")
    print("2 - ARC puzzle")
    print("3 - Bin Packing")
    print("4 - DTSP")
    print("5 - Baldwin Effect experiment")
    mode_choice = input("Enter your choice (1/2/3/4/5): ").strip()
    
    # Mutation and adaptive‑fitness policies are set directly in code – no interactive prompt.
    
    if mode_choice == "2":
        GA_MODE = "ARC"
        json_path = input("Enter path to ARC JSON file: ").strip()
        try:
            with open(json_path) as f:
                data = json.load(f)
            num_examples = len(data['train'])
            print(f"\nFound {num_examples} training examples:")
            for i, example in enumerate(data['train']):
                input_grid = example['input']
                print(f"{i+1}. Input size: {len(input_grid)}x{len(input_grid[0])}")
            example_choice = int(input(f"\nSelect example (1-{num_examples}): ")) - 1
            if example_choice < 0 or example_choice >= num_examples:
                print("Invalid choice, using first example")
                example_choice = 0
            selected_example = data['train'][example_choice]
            GA_ARC_INPUT_GRID = selected_example['input']
            GA_ARC_TARGET_GRID = selected_example['output']
            input_np = np.array(GA_ARC_INPUT_GRID)
            target_np = np.array(GA_ARC_TARGET_GRID)
            if len(GA_ARC_INPUT_GRID) != len(GA_ARC_TARGET_GRID) or len(GA_ARC_INPUT_GRID[0]) != len(GA_ARC_TARGET_GRID[0]):
                raise ValueError("Input and target grids must have the same dimensions")
        except Exception as e:
            print(f"Error loading ARC puzzle: {e}")
            exit(1)
    elif mode_choice == "3":
        GA_MODE = "BINPACKING"
        file_path = input("Enter path to BINPACKING file (default 'binpack1.txt'): ").strip()
        if not file_path:
            file_path = "binpack1.txt"
        try:
            with open(file_path) as f:
                lines = f.read().strip().splitlines()
            num_problems = int(lines[0].strip())
            bp_problems = []
            idx = 1
            for p in range(num_problems):
                problem_name = lines[idx].strip()
                idx += 1
                parts = lines[idx].strip().split()
                idx += 1
                capacity = int(parts[0])
                num_items = int(parts[1])
                optimal = int(parts[2])
                items = [int(lines[idx + i].strip()) for i in range(num_items)]
                idx += num_items
                bp_problems.append({
                    "name": problem_name,
                    "capacity": capacity,
                    "num_items": num_items,
                    "optimal": optimal,
                    "items": items
                })
            print(f"Found {len(bp_problems)} bin packing problems.")
            # --- let the user select a single problem index ----
            chosen_index_str = input(f"Select problem index (1‑{len(bp_problems)}), default 1: ").strip()
            if not chosen_index_str:
                chosen_indices = [0]
            else:
                try:
                    idx = int(chosen_index_str) - 1
                    if idx < 0 or idx >= len(bp_problems):
                        print("Invalid index, defaulting to 1.")
                        idx = 0
                except ValueError:
                    print("Invalid input, defaulting to 1.")
                    idx = 0
                chosen_indices = [idx]
            for index in chosen_indices:
                bp_instance = bp_problems[index]
                print(f"\nSolving bin packing problem {bp_instance['name']}: capacity {bp_instance['capacity']}, {bp_instance['num_items']} items, optimal bins {bp_instance['optimal']}")
                global BP_ITEMS, BP_CAPACITY, BP_OPTIMAL
                BP_ITEMS = bp_instance["items"]
                BP_CAPACITY = bp_instance["capacity"]
                BP_OPTIMAL = bp_instance["optimal"]
                # Initialize population for this problem
                population = [GAIndividual() for _ in range(GA_POPSIZE)]
                buffer = []
                best_solution = None
                best_fitness_list = []
                # ---- dynamic mutation‑rate tracking for Bin‑Packing ----
                mutation_rate_history = []
                GA_DYNAMIC_MUTRATE = GA_MUTATIONRATE   # reset to global default
                # --- Early-stopping trackers ---
                best_ever_fitness = float("inf")
                no_improve_count  = 0
                start_time = timeit.default_timer()
                #consecutive_ones = 0  # Count consecutive generations with best fitness equal to 1
                GEN_AVG_INDIV_MUT_HISTORY.clear()
                GEN_AVG_G_HISTORY.clear()
                for generation in range(50):
                    CURRENT_GENERATION = generation  # keep global counter in sync
                    tick_start = timeit.default_timer()
                    # compute µ(t) for current generation
                    GA_DYNAMIC_MUTRATE = compute_mutation_rate(generation, no_improve_count)
                    for ind in population:
                        ind.calculate_fitness_binpacking()

                    # --- adaptive fitness augmentation (AGE / NOVELTY / NONE) ---
                    apply_adaptive_fitness(population)

                    # re‑sort after fitness tweak
                    sort_population(population)
                    best_fitness = population[0].fitness
                    # ── global‑best bookkeeping & patience counter ─────────────────────
                    if best_fitness < best_ever_fitness:
                        best_ever_fitness = best_fitness
                        best_solution     = population[0].repr
                        no_improve_count  = 0
                    else:
                        no_improve_count += 1
                        best_fitness      = best_ever_fitness   # keep log monotonic
                    best_fitness_list.append(best_ever_fitness)
                    tick_end = timeit.default_timer()
                    tick_duration = tick_end - tick_start
                    total_elapsed = tick_end - start_time
                    print(f"Problem {bp_instance['name']} Gen {generation}: Best fitness = {best_fitness} (Tick: {tick_duration:.4f}s, Total: {total_elapsed:.4f}s)")
                    print(f"           µ(t) = {GA_DYNAMIC_MUTRATE:.3f}  |  indiv‑policy: {GA_INDIV_MUT_POLICY}")
                    if GEN_AVG_INDIV_MUT_HISTORY:
                        print(f"           µ_indiv_avg(t) = {GEN_AVG_INDIV_MUT_HISTORY[-1]:.3f}")
                    mutation_rate_history.append(GA_DYNAMIC_MUTRATE)
                    # Early stopping if no improvement
                    if no_improve_count >= GA_NO_IMPROVEMENT_LIMIT:
                        print(f"No improvement for {GA_NO_IMPROVEMENT_LIMIT} generations – stopping early.")
                        break
                    buffer = []
                    mate(population, buffer, GEN_AVG_INDIV_MUT_HISTORY)
                    population = buffer
                    population = apply_aging(population)
                # End of GA run for the current problem; use the best solution from the final population
                #best_solution = population[0].repr
                final_bins = []
                # Compute final bin distribution using First-Fit
                for i in best_solution:
                    size = BP_ITEMS[i]
                    placed = False
                    for b in final_bins:
                        if sum(BP_ITEMS[j] for j in b) + size <= BP_CAPACITY:
                            b.append(i)
                            placed = True
                            break
                    if not placed:
                        final_bins.append([i])
                # Print final results in the requested format
                total_runtime = total_elapsed
                print(f"\nResults for {bp_instance['name']}:")
                print(f"Bins used: {len(final_bins)}")
                print(f"Theoretical minimum: {BP_OPTIMAL}")
                deviation = len(final_bins) - BP_OPTIMAL
                print(f"Deviation from optimal: {deviation} bins")
                print(f"Runtime: {total_runtime:.4f}s")
                print("Bin details:")
                # Print bin details sequentially starting from Bin 1
                for idx, b in enumerate(final_bins, start=1):
                    sizes = [BP_ITEMS[i] for i in b]
                    total = sum(sizes)
                    print(f"Bin {idx}: {sizes}  (Total: {total}/{BP_CAPACITY})")
                # ---- plot µ(t) curve for this Bin‑Packing run ----
                if SHOW_PLOTS and mutation_rate_history:
                    plt.figure(figsize=(10, 4))
                    plt.plot(mutation_rate_history, label="µ(t)")
                    plt.xlabel("generation")
                    plt.ylabel("mutation rate")
                    plt.title("Mutation rate per generation")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                if SHOW_PLOTS and mutation_rate_history and GEN_AVG_INDIV_MUT_HISTORY:
                    plt.figure(figsize=(10, 4))
                    plt.plot(mutation_rate_history, label="population µ_pop(t)")
                    plt.plot(GEN_AVG_INDIV_MUT_HISTORY, label="avg individual µ_indiv(t)")
                    plt.xlabel("generation")
                    plt.ylabel("mutation rate")
                    plt.title("Population vs Individual mutation rate")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

                # ----- Convergence curves (fitness & auxiliary reward) -----
                if SHOW_PLOTS and best_fitness_list:
                    plt.figure(figsize=(10, 4))
                    plt.plot(best_fitness_list, label="best fitness")
                    plt.xlabel("generation")
                    plt.ylabel("fitness (bins – optimal)")
                    plt.title("Bin Packing – best fitness per generation")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

                if SHOW_PLOTS and GEN_AVG_G_HISTORY:
                    plt.figure(figsize=(10, 4))
                    plt.plot(GEN_AVG_G_HISTORY, label="⟨g(x,t)⟩")
                    plt.xlabel("generation")
                    plt.ylabel("auxiliary reward g(x,t)")
                    plt.title(f"{GA_ADAPT_FIT_POLICY} reward per generation")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

            exit(0)
        except Exception as e:
            print("Error loading bin packing file:", e)
            exit(1)
    elif mode_choice == "4":
        GA_MODE = "DTSP"
        cities_path = input("Enter path to cities.csv: ").strip()
        load_dtsp_data(cities_path)
        run_dtsp(cities_path)
        return   # DTSP pipeline finished
        # ---------- Task 5: Exploration vs. Exploitation Explanation ----------
        # The algorithm balances exploration and exploitation as follows:
        # • Exploration: Random initialization, mutation, and varied crossover operators introduce diversity
        #    and allow the search to explore new regions of the solution space.
        # • Exploitation: Sorting, elitism, and selecting parents based on the chosen selection method
        #    ensure that the best solutions are propagated and refined over generations.
    elif mode_choice == "5":
        run_baldwin_experiment()
        return
    else:
        GA_MODE = "STRING"
        print("Select fitness heuristic:")
        print("1 - ORIGINAL (sum of differences)")
        print("2 - LCS-based")
        fitness_choice = input("Enter your choice (1/2): ").strip()
        
        if fitness_choice == "1":
            GA_FITNESS_HEURISTIC = "ORIGINAL"
        elif fitness_choice == "2":
            GA_FITNESS_HEURISTIC = "LCS"
        else:
            print("Invalid choice, defaulting to ORIGINAL")
            GA_FITNESS_HEURISTIC = "ORIGINAL"
        print("Select crossover operator:")
        print("1 - SINGLE")
        print("2 - TWO")
        print("3 - UNIFORM")
        print("4 - TRIVIAL")
        print("5 - GRID")
        choice = input("Enter your choice (1/2/3/4/5): ").strip()
        
        if choice == "1":
            GA_CROSSOVER_OPERATOR = "SINGLE"
        elif choice == "2":
            GA_CROSSOVER_OPERATOR = "TWO"
        elif choice == "3":
            GA_CROSSOVER_OPERATOR = "UNIFORM"
        elif choice == "4":
            GA_CROSSOVER_OPERATOR = "TRIVIAL"
        elif choice == "5":
            GA_CROSSOVER_OPERATOR = "GRID"
        else:
            print("Invalid choice, defaulting to SINGLE")
            GA_CROSSOVER_OPERATOR = "SINGLE"
        print("Select parent selection method:")
        print("1 - RWS + Linear Scaling")
        print("2 - SUS + Linear Scaling")
        print("3 - Deterministic Tournament (K)")
        print("4 - Non-deterministic Tournament (P, K)")
        print("5 - Original (Random from top half)")
        sel_choice = input("Enter your choice (1/2/3/4/5): ").strip()
        if sel_choice == "1":
            GA_PARENT_SELECTION_METHOD = "RWS"
        elif sel_choice == "2":
            GA_PARENT_SELECTION_METHOD = "SUS"
        elif sel_choice == "3":
            GA_PARENT_SELECTION_METHOD = "TournamentDet"
        elif sel_choice == "4":
            GA_PARENT_SELECTION_METHOD = "TournamentStoch"
        elif sel_choice == "5":
            GA_PARENT_SELECTION_METHOD = "Original"
        else:
            print("Invalid choice, defaulting to RWS")
            GA_PARENT_SELECTION_METHOD = "RWS"
        try:
            k_val = int(input("Enter tournament parameter K (default 5): ").strip())
            GA_TOURNAMENT_K = k_val
        except:
            GA_TOURNAMENT_K = 5
        try:
            p_val = float(input("Enter tournament probability P (default 0.8): ").strip())
            GA_TOURNAMENT_P = p_val
        except:
            GA_TOURNAMENT_P = 0.8
        try:
            age_val = int(input("Enter maximum age (generations) for aging (default 10): ").strip())
            GA_MAX_AGE = age_val
        except:
            GA_MAX_AGE = 10

    # (Individual mutation policy prompt moved to top)
    random.seed(time.time())
    start_time = timeit.default_timer()
    population = init_population()
    buffer = []
    GEN_AVG_INDIV_MUT_HISTORY.clear()
    best_fitness_list = []
    avg_fitness_list = []
    worst_fitness_list = []
    fitness_distributions = []
    generation = 0
    best_solution = None
    # --- Early-stopping trackers ---
    best_ever_fitness = float("inf")
    no_improve_count  = 0
    # --- Monotone best-so-far tracking ---
    monotone_best_history = []
    global_best_fitness = None
    global_best_individual = None
    # --- global best‑so‑far bookkeeping (monotone decrease expected) ---
    best_ever = None                 # stores the chromosome / path pair
    best_ever_fitness = float('inf') # minimization: lower is better
    while generation < GA_MAXITER:
        CURRENT_GENERATION = generation
        tick_start = timeit.default_timer()
        for ind in population:
            if GA_MODE == "DTSP":
                ind.calculate_fitness_dtsp()
            elif GA_FITNESS_HEURISTIC == "ORIGINAL":
                ind.calculate_fitness()
            else:
                ind.calculate_fitness_lcs()
        CURRENT_GENERATION = 0
        apply_adaptive_fitness(population)      # NEW
        sort_population(population)
        fitness_values = [ind.fitness for ind in population]
        fitness_distributions.append(fitness_values.copy())
        best = population[0]
        best_fitness = best.fitness
        best_index = 0
        worst_fitness = population[-1].fitness
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        # update global best‑so‑far if this generation improved it
        if best_fitness < best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever = copy.deepcopy(best)
        # --- Track global best so far (monotone) ---
        if generation == 0:
            global_best_fitness = best_fitness   # first generation best
            global_best_individual = copy.deepcopy(population[best_index])
        else:
            if best_fitness < global_best_fitness:
                global_best_fitness = best_fitness
                global_best_individual = copy.deepcopy(population[best_index])
        # Store the monotone series for plotting
        monotone_best_history.append(global_best_fitness)
        # ── global‑best bookkeeping & patience counter ─────────────────────
        if best_fitness < best_ever_fitness:
            best_solution     = population[0].repr
            no_improve_count  = 0
        else:
            no_improve_count += 1
            best_fitness      = best_ever_fitness   # keep log monotonic
        best_fitness_list.append(best_ever_fitness)
        avg_fitness_list.append(avg_fitness)
        worst_fitness_list.append(worst_fitness)
        tick_end = timeit.default_timer()
        tick_duration = tick_end - tick_start
        total_elapsed = tick_end - start_time
        print(f"Gen {generation}: Best‑so‑far = {best_ever.repr if hasattr(best_ever, 'repr') else best_ever} (Fitness = {best_ever_fitness})")
        print(f"           µ_pop(t) = {GA_DYNAMIC_MUTRATE:.3f}  |  μ‑policy: {GA_INDIV_MUT_POLICY}")
        if GEN_AVG_INDIV_MUT_HISTORY:
            print(f"           µ_indiv_avg(t) = {GEN_AVG_INDIV_MUT_HISTORY[-1]:.3f}")
        if GEN_AVG_G_HISTORY:
            print(f"           ⟨g(x,t)⟩ = {GEN_AVG_G_HISTORY[-1]:.3f}  |  g‑policy: {GA_ADAPT_FIT_POLICY}")
        if population[0].fitness == 0:
            best_solution = population[0].repr
            print(f"\n*** Converged after {generation + 1} generations ***")
            break
        if no_improve_count >= GA_NO_IMPROVEMENT_LIMIT:
            print(f"No improvement for {GA_NO_IMPROVEMENT_LIMIT} generations – stopping early.")
            break
        buffer.clear()
        mate(population, buffer, GEN_AVG_INDIV_MUT_HISTORY)
        # --- Elitism: ensure the best-so-far survives ---
        population[random.randrange(len(population))] = copy.deepcopy(global_best_individual)
        population, buffer = buffer, population
        population = apply_aging(population)
        generation += 1
    if best_solution is None and GA_MODE == "ARC":
        print("\n*** Final Best Attempt ***")
        plot_grids(np.array(GA_ARC_INPUT_GRID), np.array(GA_ARC_TARGET_GRID), np.array(population[0].repr))
        print(f"Matching Cells: {len(GA_ARC_INPUT_GRID)*len(GA_ARC_INPUT_GRID[0]) - population[0].fitness}/{len(GA_ARC_INPUT_GRID)*len(GA_ARC_INPUT_GRID[0])}")
    elif GA_MODE == "ARC":
        solution_np = np.array(best_solution) if best_solution else None
        plot_grids(np.array(GA_ARC_INPUT_GRID), np.array(GA_ARC_TARGET_GRID), solution_np)
        if best_solution is None:
            print(f"\nMatching Cells: {len(GA_ARC_INPUT_GRID)*len(GA_ARC_INPUT_GRID[0]) - population[0].fitness}/{len(GA_ARC_INPUT_GRID)*len(GA_ARC_INPUT_GRID[0])}")
    plt.figure(figsize=(10, 6))
    generations = list(range(len(monotone_best_history)))
    plt.plot(generations, monotone_best_history, label="Best‑so‑far")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Behavior per Generation")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.boxplot(fitness_distributions, showfliers=True)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Box Plot of Fitness per Generation")
    plt.grid(True)
    plt.show()
    if best_fitness_list and GEN_AVG_INDIV_MUT_HISTORY:
        plt.figure(figsize=(10, 4))
        plt.plot(best_fitness_list, label="Best fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Best Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    if GEN_AVG_G_HISTORY:
        plt.figure(figsize=(10,4))
        plt.plot(GEN_AVG_G_HISTORY, label="⟨g(x,t)⟩")
        plt.xlabel("generation"); plt.ylabel("auxiliary reward g(x,t)")
        plt.title(f"{GA_ADAPT_FIT_POLICY} reward per generation")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
        plt.figure(figsize=(10, 4))
        plt.plot(best_fitness_list, label="Best fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Best Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    if avg_fitness_list and GEN_AVG_INDIV_MUT_HISTORY:
        plt.figure(figsize=(10, 4))
        plt.plot(avg_fitness_list, label="Average fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Average Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    if worst_fitness_list and GEN_AVG_INDIV_MUT_HISTORY:
        plt.figure(figsize=(10, 4))
        plt.plot(worst_fitness_list, label="Worst fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Worst Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    # Plot population and average individual mutation rate
    if avg_fitness_list and GEN_AVG_INDIV_MUT_HISTORY:
        plt.figure(figsize=(10, 4))
        plt.plot(avg_fitness_list, label="avg fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Average Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    if best_fitness_list and GEN_AVG_INDIV_MUT_HISTORY:
        plt.figure(figsize=(10, 4))
        plt.plot(best_fitness_list, label="Best fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Best Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    if worst_fitness_list and GEN_AVG_INDIV_MUT_HISTORY:
        plt.figure(figsize=(10, 4))
        plt.plot(worst_fitness_list, label="Worst fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Worst Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    if best_fitness_list and GEN_AVG_INDIV_MUT_HISTORY:
        plt.figure(figsize=(10, 4))
        plt.plot(best_fitness_list, label="Best fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Best Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    if avg_fitness_list and GEN_AVG_INDIV_MUT_HISTORY:
        plt.figure(figsize=(10, 4))
        plt.plot(avg_fitness_list, label="Average fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Average Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    if worst_fitness_list and GEN_AVG_INDIV_MUT_HISTORY:
        plt.figure(figsize=(10, 4))
        plt.plot(worst_fitness_list, label="Worst fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Worst Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    # Plot population vs individual mutation rate
    if avg_fitness_list and GEN_AVG_INDIV_MUT_HISTORY:
        plt.figure(figsize=(10, 4))
        plt.plot(avg_fitness_list, label="avg fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Average Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    if best_fitness_list and GEN_AVG_INDIV_MUT_HISTORY:
        plt.figure(figsize=(10, 4))
        plt.plot(best_fitness_list, label="Best fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Best Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    if worst_fitness_list and GEN_AVG_INDIV_MUT_HISTORY:
        plt.figure(figsize=(10, 4))
        plt.plot(worst_fitness_list, label="Worst fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Worst Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    if len(best_fitness_list) and len(GEN_AVG_INDIV_MUT_HISTORY):
        plt.figure(figsize=(10, 4))
        plt.plot(best_fitness_list, label="Best fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Best Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    # Population vs Individual mutation rate plot
    if len(avg_fitness_list) and len(GEN_AVG_INDIV_MUT_HISTORY):
        plt.figure(figsize=(10, 4))
        plt.plot(avg_fitness_list, label="avg fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Average Fitness per Generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    # The actual requested plot:
    if len(monotone_best_history) and len(GEN_AVG_INDIV_MUT_HISTORY):
        plt.figure(figsize=(10, 4))
        plt.plot(monotone_best_history, label="population µ_pop(t)")
        plt.plot(GEN_AVG_INDIV_MUT_HISTORY, label="avg individual µ_indiv(t)")
        plt.xlabel("generation")
        plt.ylabel("mutation rate")
        plt.title("Population vs Individual mutation rate")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    # ---------- Task 5: Exploration vs. Exploitation Explanation ----------
    # The algorithm balances exploration and exploitation as follows:
    # • Exploration: Random initialization, mutation, and varied crossover operators introduce diversity
    #    and allow the search to explore new regions of the solution space.
    # • Exploitation: Sorting, elitism, and selecting parents based on the chosen selection method
    #    ensure that the best solutions are propagated and refined over generations.

    # Add exit statement at the end of main()
    sys.exit(0)

if __name__ == "__main__":
    main()
 