def read_movies(filename):
    """
    Returnerer dictionary med filmer.
    """
    movies = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            movie_id = parts[0]
            movie_title = parts[1]
            movie_rating = float(parts[2])
            movies[movie_id] = {'title': movie_title, 'rating': movie_rating}
    return movies 

def read_actors(filename):
    """
    Returnerer dictionary med skuespillere.
    """
    actors = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            actor_id = parts[0]
            actor_name = parts[1]
            movie_ids = set(parts[2:])
            actors[actor_id] = {'name': actor_name, 'movies': movie_ids}
    return actors


def build_graph(movies, actors):
    """
    Bygger grafen med skuespillere som noder og filmer som kantene mellom dem.

    1.  Lager først en dictionary der man legger filmene som nøkler og set() med actors som verdier.

    2.  Lager nodene som er skuespillerne med hver sine kanter
    """
    # 1.
    film_to_actors = {}
    for actor_id, details in actors.items():
        for movie_id in details['movies']:
            if movie_id not in film_to_actors:
                film_to_actors[movie_id] = set()
            film_to_actors[movie_id].add(actor_id)

    # 2.
    graph = {'movies': movies, 'actors': {}, 'edges': {}}
    for movie_id, actors_in_movie in film_to_actors.items():
        # fra set til liste
        actors_list = list(actors_in_movie)
        for i in range(len(actors_list)):
            actor1 = actors_list[i]
            #sjekker om skuespiller er lagt til i 'actors' allerede. Hvis ikke blir de lagt til
            if actor1 not in graph['actors']:
                graph['actors'][actor1] = {'name': actors[actor1]['name'], 'movies': actors[actor1]['movies'], 'edges': set()}
            # Lager kanter, mellom skuespillerne og legger dem til hvis de ikke har blitt lagt til i 'actors enda. Bruker +1 for å ikke lage kant med seg selv
            # Actor1 for altså en kant, med alle skuespillerne undeer samme film
            for j in range(i + 1, len(actors_list)):
                actor2 = actors_list[j]
                if actor2 not in graph['actors']:
                    graph['actors'][actor2] = {'name': actors[actor2]['name'], 'movies': actors[actor2]['movies'], 'edges': set()}
                sorted_actors = tuple(sorted([actor1, actor2]))
                # Legger tilkanten mellom i egen dicitonary 'edges'
                if sorted_actors not in graph['edges']:
                    graph['edges'][sorted_actors] = set()
                # 'edges' er en dictionary der nøklene er tupler av skuespillere og filmer som verdier
                graph['edges'][sorted_actors].add(movie_id)
                # Legger til kanten inn i hver skuespiller
                graph['actors'][actor1]['edges'].add((actor2, tuple(graph['edges'][sorted_actors])))
                graph['actors'][actor2]['edges'].add((actor1, tuple(graph['edges'][sorted_actors])))

    return graph

def bfs_shortest_path(graph, start_id, goal_id):
    """
    Bruker bfs (Breadth first search)

    Finner korteste veien ved å sette opp masse paths og den som finner frem først vil være den korteste veien.
    """
    queue = [([start_id], [])]  # Kø oav tupler der tuplene er består av paths og filmer som skal sjekkes. feks [([actor1, actor2, actor5], [film1_2, film2_5]), ([actor1, actor3, actor6], [film1_3, film3_6])]
    visited = set()

    # Går hetl til queue er tom
    while queue:
        # FIFO
        path, movies = queue.pop(0)
        node = path[-1]
        
        # sjekker om man har kommet frem til målet
        if node == goal_id:
            return path, movies

     
        if node not in visited:
            #legger til i visited slik at man ikke trenger å gå gjennom actoren en gang til
            visited.add(node)
            # går gjennom alle skuespillerne koble til skuespilleren man ser på akkurat nå og legger dem til i køen slik at disse også blir sett neste gang.
            for connected_actor, movie_ids in graph['actors'][node]['edges']:
                # Hvis skuespilleren har vært besøkt så har den allerede vært gjennom køen og alt dette
                if connected_actor not in visited:
                    # 'Uknown Movie' kommer hvis man ikke finner filmen i movie_ids. Standardverdien. Denne fikke jeg fra chatgpt
                    movie_titles = [graph['movies'].get(movie_id, {'title': 'Unknown Movie'})['title'] for movie_id in movie_ids]
                    new_path = path + [connected_actor]
                    new_movies = movies + movie_titles
                    # Legger til pathen som den har tatt feks hvis du sjekker
                    queue.append((new_path, new_movies))

    return None


def print_shortest_path(graph, start_id, end_id):
    """
    Prosedyre som brukes for å skrive ut resultatet av BFS-søket
    """
    result = bfs_shortest_path(graph, start_id, end_id)
    if result:
        path, movies = result
        print(f"Path from {graph['actors'][start_id]['name']} to {graph['actors'][end_id]['name']}:")
        for i in range(len(path) - 1):
            print(f"{graph['actors'][path[i]]['name']} ===[ {movies[i]} ] ===> {graph['actors'][path[i + 1]]['name']}")
    else:
        print("No path found.")


def count_nodes(graph, type):
    """
    Teller antall noder 
    """
    return len(graph[type])

def count_edges(graph):
    """
    Teller antall unike kanter i grafen. Kanter som i actor1 og actor2 som har spill i samme film
    """
    return len(graph['edges'])




# Hovedprogram
filename_movies = 'movies.tsv'
filename_actors = 'actors.tsv'
movies = read_movies(filename_movies)
actors = read_actors(filename_actors)
graph = build_graph(movies, actors)

# Tell noder og kanter
print("Antall filmer:", count_nodes(graph, 'movies'))
print("Antall skuespillere:", count_nodes(graph, 'actors'))
print("Antall unike kanter (samarbeid):", count_edges(graph))
# Donald Glover til Jeremy Irons
print_shortest_path(graph, 'nm2255973', 'nm0000460')

# Scarlett Johansson til Emma Mackey
print_shortest_path(graph, 'nm0424060', 'nm8076281')

# Carrie Coon til Julie DelpyT
print_shortest_path(graph, 'nm4689420', 'nm0000365')

# Christian Bale til Lupita Nyong’o
print_shortest_path(graph, 'nm0000288', 'nm2143282')

# Tuva Novotny til Michael K. Williams
print_shortest_path(graph, 'nm0637259', 'nm0931324')

