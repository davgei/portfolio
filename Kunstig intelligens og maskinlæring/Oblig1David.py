import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
import csv
import time

# %matplotlib inline
np.random.seed(57)
#Map of Europe
europe_map = plt.imread('map.png')

#Lists of city coordinates
city_coords = {
    "Barcelona": [2.154007, 41.390205], "Belgrade": [20.46, 44.79], "Berlin": [13.40, 52.52], 
    "Brussels": [4.35, 50.85], "Bucharest": [26.10, 44.44], "Budapest": [19.04, 47.50],
    "Copenhagen": [12.57, 55.68], "Dublin": [-6.27, 53.35], "Hamburg": [9.99, 53.55], 
    "Istanbul": [28.98, 41.02], "Kyiv": [30.52, 50.45], "London": [-0.12, 51.51], 
    "Madrid": [-3.70, 40.42], "Milan": [9.19, 45.46], "Moscow": [37.62, 55.75],
    "Munich": [11.58, 48.14], "Paris": [2.35, 48.86], "Prague": [14.42, 50.07],
    "Rome": [12.50, 41.90], "Saint Petersburg": [30.31, 59.94], "Sofia": [23.32, 42.70],
    "Stockholm": [18.06, 60.33], "Vienna": [16.36, 48.21], "Warsaw": [21.02, 52.24]}

#Helper code for plotting plans
#First, visualizing the cities.
import csv
with open("european_cities.csv", "r") as f:
    data = list(csv.reader(f, delimiter=';'))
    cities = data[0]

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(europe_map, extent=[-14.56, 38.43, 37.697 + 0.3, 64.344 + 2.0], aspect="auto")

# Map (long, lat) to (x, y) for plotting
for city, location in city_coords.items():
    x, y = (location[0], location[1])
    plt.plot(x, y, 'ok', markersize=5)
    plt.text(x, y, city, fontsize=12)

    #A method you can use to plot your plan on the map.
def plot_plan(city_order):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(europe_map, extent=[-14.56, 38.43, 37.697 + 0.3, 64.344 + 2.0], aspect="auto")

    # Map (long, lat) to (x, y) for plotting
    for index in range(len(city_order) - 1):
        current_city_coords = city_coords[city_order[index]]
        next_city_coords = city_coords[city_order[index+1]]
        x, y = current_city_coords[0], current_city_coords[1]
        #Plotting a line to the next city
        next_x, next_y = next_city_coords[0], next_city_coords[1]
        plt.plot([x, next_x], [y, next_y])

        plt.plot(x, y, 'ok', markersize=5)
        plt.text(x, y, index, fontsize=12)
    #Finally, plotting from last to first city
    first_city_coords = city_coords[city_order[0]]
    first_x, first_y = first_city_coords[0], first_city_coords[1]
    plt.plot([next_x, first_x], [next_y, first_y])
    #Plotting a marker and index for the final city
    plt.plot(next_x, next_y, 'ok', markersize=5)
    plt.text(next_x, next_y, index+1, fontsize=12)
    plt.show()

#Example usage of the plotting-method.
plan = list(city_coords.keys()) # Gives us the cities in alphabetic order



#Exhaustive search - oppgave 1

#lettere å  lese distanser med denne oppslagstabellen
with open("european_cities.csv", "r", encoding="utf-8") as f:
    rows = list(csv.reader(f, delimiter=";"))

cities = rows[0]              # første linje = bynavn
distances = {c: {} for c in cities}  # tom dictionary

#lettere å lese distanser med denne dictionarien. Fungere slik : distances[by1][by2] -> distnase mellom byene
for i, row in enumerate(rows[1:]):  
    city_a = cities[i]
    for j, value in enumerate(row):
        city_b = cities[j]
        distances[city_a][city_b] = float(value)

def route_distance(route):
    total = 0
    #går gjennom alle byene i ruta
    for i in range(len(route) - 1):
        total += distances[route[i]][route[i+1]] #lengde til neste by i ruta
    total += distances[route[-1]][route[0]]  # tilbake til start
    return total

def exhaustive(city_list):
    #alle ruter som finnes med de forskjellige x antall byene

    best_route = None
    best_len =  float("inf")

    for route in itertools.permutations(city_list):
        current_len= route_distance(route)
        if current_len < best_len:
            best_len = current_len
            best_route = route
            
    return best_len, list(best_route)

def timed_exhaustive(city_list):
    t0 = time.perf_counter()
    best_len, best_route = exhaustive(city_list)
    t1 = time.perf_counter()
    return best_len, best_route, (t1 - t0)

cities_subset_6 = random.sample(list(city_coords.keys()), 6) #6 forskjellige tilfeldige byer
cities_subset_10 = cities[:10] #de 10 frøste byene
valid_cities = [c for c in cities if c in city_coords] #sjekker opp csv mot city_coords slik at man ikke får error hvis noe ikke samsvaerr

# 6 tilfeldige byer
cities_subset_6 = random.sample(valid_cities, 6)
best_len_6, best_route_6, sec_6 = timed_exhaustive(cities_subset_6) 
print("Exhaustive search 6 byer:", round(best_len_6, 2), "km,", 
      "tid:", round(sec_6, 3), "s,", 
      "rute:", best_route_6)
plot_plan(best_route_6)

# første 10 byene fra CSV 
cities_subset_10 = [c for c in cities[:10] if c in city_coords]
best_len_10, best_route_10, sec_10 = timed_exhaustive(cities_subset_10)
print("Exhaustive search 10 byer:", round(best_len_10, 2), "km,", 
      "tid:", round(sec_10, 3), "s,", 
      "rute:", best_route_10)
plot_plan(best_route_10)



#hill climber oppgave 2
def hill_climber(route, iterations):
    
    current_route = random.sample(route, len(route))
    current_len = route_distance(current_route)

    best_route = current_route[:]
    best_len = current_len


    for i in range(iterations):
        new_route = current_route[:]
        a, b = random.sample(range(len(current_route)), 2) # to tilfeldige tall a og b
        new_route[a], new_route[b] = new_route[b], new_route[a] #by a og b bytter plass
        new_len = route_distance(new_route)
        if new_len < current_len:
            current_len, current_route = new_len, new_route
            if best_len > new_len:
                best_len, best_route  = new_len, new_route

    return best_route, best_len


def run_hill_climber(city_subset, iterations=10000, runs=20, label=""):
    lengths = []
    routes = []

    for _ in range(runs):
        r, l = hill_climber(city_subset, iterations)
        routes.append(r)
        lengths.append(l)

    lengths = np.array(lengths, dtype=float)
    best = float(lengths.min())
    worst = float(lengths.max())
    mean = float(lengths.mean())
    std = float(lengths.std(ddof=0))

    print("\nHill Climber (", label, ") best:", round(best, 2),
      "worst:", round(worst, 2),
      "mean:", round(mean, 2),
      "std:", round(std, 2))


    # plott den beste ruten
    plot_plan(routes[int(lengths.argmin())])


#10 byer
run_hill_climber(cities_subset_10, iterations=10000, runs=20, label="10 byer")

#24 byer 
cities_subset_24 = [c for c in cities if c in city_coords]  # sikrer riktige bynavn
run_hill_climber(cities_subset_24, iterations=10000, runs=20, label="24 byer")


# GA  opgave 3

def make_permutation(cities):
    #lager en permutasjon av byene
    return random.sample(cities, len(cities))

def find_parent(population, k):
    candidates = random.sample(population, k)
    return min(candidates, key=route_distance) #returnerer rute med kortest distanse

def ox_crossover(p1,p2):
    # ettersom bare rekkefølge har noe å si i TSP og ikke hvilken plass i lista, ser jeg bort ifra hvor segmentene skal være i lista.
    a = random.randint(0, len(p1) - len(p1)//2)
    b = a + len(p1)//2
    c1 = p1[a:b]
    c2 = p2[a:b]  
    for city in p2:
        if city not in c1:     
            c1.append(city)
    for city in p1:
        if city not in c2:     
            c2.append(city)
    return c1, c2


def mutate_child(child):
    a, b = random.sample(range(len(child)), 2)
    child[a], child[b] = child[b], child[a]




def GA(cities, p_size, generations, cx_rate, mut_rate, k_tourn):
    
    
    population = [make_permutation(cities) for _ in range(p_size)] #liste med forskjellige ruter
    population_length = [route_distance(route) for route in population]
    best_per_gen = [] #viser hvilken distanse på ruta som var best per generasjon. 

    for _ in range(generations):
        elite = population[population_length.index(min(population_length))]
        new_population=[elite]
        best_per_gen.append(route_distance(elite))

        for _ in range(len(population)//2): #delt på 2 fordi jeg bare skal ha halvparten foreldre
            parent_a, parent_b = [find_parent(population, k_tourn) for _ in range(2)]
            
            #OX order crossover 
            if random.random() < cx_rate:
                c1, c2 = ox_crossover(parent_a, parent_b) # bruk OX
            else:
                c1, c2 = parent_a[:], parent_b[:] #kopier foreldrene direkte

            #mutasjon
            if random.random() < mut_rate: #muterer ikke allti 
                mutate_child(c1) 
            if random.random() < mut_rate:
                mutate_child(c2)

            new_population.extend([c1, c2])
        
        population = new_population[:p_size]#fordi man legger til elite på starten. Så man ender opp med 100 istedenfor 101.
        population_length = [route_distance(route) for route in population]
    
    elite = population[population_length.index(min(population_length))]
    best_per_gen.append(route_distance(elite))

    return elite, route_distance(elite), best_per_gen


pop_sizes = [30, 100, 300]
G = 300 # generasjoner
RUNS = 20# antall kjøringer (oppgavespesifikt)

def run_many(cities_list, population_size, runs=RUNS, G=G):
    best_lengths = []
    curves = []
    t0 = time.perf_counter()
    best_routes = []
    for _ in range(runs):
        elite, L, hist = GA(cities_list, p_size=population_size, generations=G, cx_rate=0.9, mut_rate=0.2, k_tourn=3)
        best_lengths.append(L)
        curves.append(np.array(hist, float))
        best_routes.append(elite)
    t1 = time.perf_counter()
    best_lengths = np.array(best_lengths, float)
    avg_curve = np.mean(np.stack(curves), axis=0)
    #returnerer i dictionary form
    return {
        "best": float(best_lengths.min()),
        "worst": float(best_lengths.max()),
        "mean": float(best_lengths.mean()),
        "std": float(best_lengths.std(ddof=0)),
        "avg_curve": avg_curve,
        "route": best_routes[int(best_lengths.argmin())],
        "seconds": t1 - t0
    }

def report(title, results_dict):
    print("\nResultater for", title)
    for population_size in pop_sizes:
        r = results_dict[population_size]
        print(
            "Populasjonsstørrelse", population_size, 
            "beste:", round(r["best"], 2), "km,", 
            "verste:", round(r["worst"], 2), "km,", 
            "gjennomsnitt:", round(r["mean"], 2), "km,", 
            "standardavvik:", round(r["std"], 2), ",", 
            "tid:", round(r["seconds"], 2), "s"
        )

    # felles plott av gjennomsnittlig beste distanse per generasjon
    plt.figure(figsize=(7,4))
    for population_size in pop_sizes:
        plt.plot(results_dict[population_size]["avg_curve"], label="Populasjkonstørrelse=" + str(population_size))
    plt.xlabel("Generasjon")
    plt.ylabel("Beste distanse (snitt over 20 kjøringer)")
    plt.title("Genetic Algorithm  " + title)
    plt.legend()
    plt.show()

    #eksempelrute for hver populasjonsstørrelse
    for population_size in pop_sizes:
        plot_plan(results_dict[population_size]["route"])

#10 byer
results_10 = {P: run_many(cities_subset_10, P) for P in pop_sizes} #lager en dictionary der man har info om kjøring for 30, 100 og 300
report("10 byer", results_10)

#24 byer
results_24 = {P: run_many(cities_subset_24, P) for P in pop_sizes} #lager en dictionary der man har info om kjøring for 30, 100 og 300
report("24 byer", results_24)