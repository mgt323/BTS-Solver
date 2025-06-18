import csv
import math
import random
import os
from typing import List, Tuple, Dict, Set


def load_synergy_map_from_csv(filepath, delimiter=';'):
    """
    Wczytuje mapę synergii z pliku CSV.
    Zwraca mapę synergii lub None jeśli wystąpi błąd.
    """
    synergy_map = {}
    if not os.path.exists(filepath):
        print(f"Błąd krytyczny: Plik mapy synergii '{filepath}' nie został znaleziony.")
        return None
    try:
        with open(filepath, mode='r', encoding='utf-8-sig') as file:
            reader = csv.reader(file, delimiter=delimiter)
            header = next(reader, None)
            if not header or len(header) < 2:
                print(f"Błąd: Nieprawidłowy nagłówek w pliku synergii: {header}")
                return None

            col_technologies = [tech.strip().upper() for tech in header[1:]]

            for row_idx, row_values in enumerate(reader):
                if not row_values or len(row_values) < len(col_technologies) + 1:
                    print(f"Ostrzeżenie: Pomijanie niekompletnego wiersza {row_idx + 2} w pliku synergii.")
                    continue

                row_tech_name = row_values[0].strip().upper()
                if not row_tech_name:
                    continue

                synergy_map[row_tech_name] = {}
                synergy_values_for_row = row_values[1:]
                for i, col_tech_name in enumerate(col_technologies):
                    try:
                        value_str = synergy_values_for_row[i].strip()
                        synergy_value = float(value_str.replace(',', '.')) if value_str else 0.0
                        synergy_map[row_tech_name][col_tech_name] = synergy_value
                    except (ValueError, IndexError):
                        synergy_map[row_tech_name][col_tech_name] = 0.0

        # Automatyczne uzupełnienie symetrycznych wartości
        all_techs_in_map = sorted(list(synergy_map.keys()))
        for tech_a in all_techs_in_map:
            for tech_b in all_techs_in_map:
                val_ab = synergy_map.get(tech_a, {}).get(tech_b)
                val_ba = synergy_map.get(tech_b, {}).get(tech_a)
                if val_ab is None and val_ba is not None:
                    synergy_map[tech_a][tech_b] = val_ba
                elif val_ba is None and val_ab is not None:
                    synergy_map[tech_b][tech_a] = val_ab
                elif val_ab is None and val_ba is None:
                    synergy_map[tech_a][tech_b] = 0.0
                    synergy_map[tech_b][tech_a] = 0.0

        print(f"Pomyślnie wczytano mapę synergii z '{filepath}'")
        return synergy_map
    except Exception as e:
        print(f"Nieoczekiwany błąd podczas wczytywania mapy synergii z '{filepath}': {e}")
        return None


class QAPProblem:
    def __init__(self, csv_filepath, loaded_synergy_map):
        self.csv_filepath = csv_filepath
        self.stations = []
        self.coordinates = []  # (longitude, latitude) - (x, y)
        self.num_stations = 0
        self.distance_matrix_D = []
        self.synergy_map_keys = set(loaded_synergy_map.keys()) if loaded_synergy_map else set()

        self._load_stations()
        if self.num_stations > 0:
            self._calculate_distance_matrix()

    def _load_stations(self):
        expected_headers = ['ID stacji', 'latitude', 'longitude', 'Technologie']
        try:
            with open(self.csv_filepath, mode='r', encoding='utf-8-sig') as file:
                reader = csv.DictReader(file, delimiter=',')

                if not reader.fieldnames:
                    print(f"Błąd: Plik stacji '{self.csv_filepath}' jest pusty lub nie zawiera nagłówków.")
                    return

                missing_headers = [h for h in expected_headers if h not in reader.fieldnames]
                if missing_headers:
                    print(
                        f"Błąd: Brakujące nagłówki w pliku '{self.csv_filepath}': {missing_headers}. Oczekiwano: {expected_headers}.")
                    return

                for row_idx, row in enumerate(reader):
                    try:
                        station_id_str = row['ID stacji'].strip()
                        longitude = float(row['longitude'].replace(',', '.').strip())
                        latitude = float(row['latitude'].replace(',', '.').strip())

                        # AKTUALIZACJA: Parsowanie listy technologii dla nowego formatu
                        # Czytnik CSV sam obsługuje cudzysłowy otaczające całe pole
                        raw_tech_string = row['Technologie'].strip()
                        station_allowed_techs_raw = [
                            tech.strip().upper() for tech in raw_tech_string.split(',') if tech.strip()
                        ]

                        if not station_allowed_techs_raw:
                            print(
                                f"Ostrzeżenie: Brak technologii dla stacji '{station_id_str}' (wiersz {row_idx + 2}). Pomijanie.")
                            continue

                        valid_station_techs = []
                        for tech in station_allowed_techs_raw:
                            if tech not in self.synergy_map_keys:
                                print(
                                    f"Ostrzeżenie: Technologia '{tech}' (stacja '{station_id_str}') nie jest w mapie synergii. Zostanie zignorowana.")
                            else:
                                valid_station_techs.append(tech)

                        if not valid_station_techs:
                            print(f"Ostrzeżenie: Brak WAŻNYCH technologii dla stacji '{station_id_str}'. Pomijanie.")
                            continue

                        self.stations.append({
                            'original_id': station_id_str, 'id': self.num_stations,
                            'x': longitude, 'y': latitude,
                            'initial_allowed_technologies': valid_station_techs
                        })
                        self.coordinates.append((longitude, latitude))
                        self.num_stations += 1
                    except (ValueError, KeyError, Exception) as e:
                        print(
                            f"Błąd przetwarzania wiersza {row_idx + 2} w '{self.csv_filepath}' ({row}): {e}. Pomijanie.")

        except FileNotFoundError:
            print(f"Błąd krytyczny: Plik stacji '{self.csv_filepath}' nie został znaleziony.")
            self.num_stations = 0
        except Exception as e:
            print(f"Błąd krytyczny podczas otwierania pliku stacji '{self.csv_filepath}': {e}")
            self.num_stations = 0

    def _calculate_distance_matrix(self):
        self.distance_matrix_D = [[0.0] * self.num_stations for _ in range(self.num_stations)]
        for i in range(self.num_stations):
            for j in range(i + 1, self.num_stations):
                coord1_x, coord1_y = self.coordinates[i]
                coord2_x, coord2_y = self.coordinates[j]
                dist = math.sqrt((coord1_x - coord2_x) ** 2 + (coord1_y - coord2_y) ** 2)
                self.distance_matrix_D[i][j] = dist
                self.distance_matrix_D[j][i] = dist

    def get_initial_candidate_assignment(self, strategy="first"):
        if self.num_stations == 0: return []
        assignment = []
        for station_data in self.stations:
            allowed_techs = station_data['initial_allowed_technologies']
            if not allowed_techs:
                raise ValueError(
                    f"Stacja ID {station_data['id']} (oryg. {station_data['original_id']}) nie ma dozwolonych technologii po walidacji z mapą synergii.")

            chosen_tech = ""
            if strategy == "first":
                chosen_tech = allowed_techs[0]
            elif strategy == "random":
                chosen_tech = random.choice(allowed_techs)
            else:
                raise ValueError(f"Nieznana strategia inicjalizacji: {strategy}")
            assignment.append(chosen_tech)
        return assignment

    def get_allowed_technologies_for_station(self, station_idx):
        if 0 <= station_idx < self.num_stations:
            return self.stations[station_idx]['initial_allowed_technologies']
        raise IndexError(f"Nieprawidłowy wewnętrzny indeks stacji: {station_idx}")


class QAPSolver:
    def __init__(self, qap_problem, synergy_map, proximity_function=None):
        self.problem = qap_problem
        self.synergy_map = synergy_map
        if proximity_function is None:
            self.proximity_function = lambda d: 1.0 / (1.0 + d) if d >= 0 else 0
        else:
            self.proximity_function = proximity_function

    def _get_synergy(self, tech1_name, tech2_name):
        if tech1_name not in self.synergy_map:
            return 0
        tech1_map = self.synergy_map.get(tech1_name, {})

        if tech2_name not in tech1_map:
            return 0
        return tech1_map.get(tech2_name, 0)

    def calculate_synergy_score(self, assignment_of_technologies):
        if self.problem.num_stations == 0: return 0.0
        if len(assignment_of_technologies) != self.problem.num_stations:
            raise ValueError("Liczba technologii w przypisaniu musi odpowiadać liczbie stacji.")

        total_synergy_score = 0.0
        for i in range(self.problem.num_stations):
            for j in range(i + 1, self.problem.num_stations):
                tech_i = assignment_of_technologies[i]
                tech_j = assignment_of_technologies[j]
                synergy_value = self._get_synergy(tech_i, tech_j)
                distance = self.problem.distance_matrix_D[i][j]
                proximity = self.proximity_function(distance)
                total_synergy_score += synergy_value * proximity
        return total_synergy_score * 2.0

    def calculate_score_delta(self, current_assignment, station_idx, new_tech):
        """
        Oblicza przyrostową zmianę wyniku synergii po zmianie technologii na jednej stacji.
        Złożoność: O(N)
        """
        old_tech = current_assignment[station_idx]

        if old_tech == new_tech:
            return 0.0

        old_contribution = 0.0
        new_contribution = 0.0

        for i in range(self.problem.num_stations):
            if i == station_idx:
                continue

            other_station_tech = current_assignment[i]
            proximity = self.proximity_function(self.problem.distance_matrix_D[station_idx][i])

            old_contribution += self._get_synergy(old_tech, other_station_tech) * proximity
            new_contribution += self._get_synergy(new_tech, other_station_tech) * proximity

        return (new_contribution - old_contribution) * 2.0


class RandomSolver:
    def __init__(self, qap_problem, synergy_map, proximity_function=None):
        self.problem = qap_problem
        self.synergy_calculator = QAPSolver(qap_problem, synergy_map, proximity_function)

    def solve(self, num_trials=1000):
        if self.problem.num_stations == 0:
            print("Nie można uruchomić RandomSolver - brak stacji w problemie.")
            return None, -float('inf')

        best_assignment = None
        best_score = -float('inf')

        print(f"\nUruchamianie RandomSolver (liczba prób: {num_trials})...")
        for i in range(num_trials):
            current_assignment = []
            for station_idx in range(self.problem.num_stations):
                station_specific_allowed_techs = self.problem.get_allowed_technologies_for_station(station_idx)
                if not station_specific_allowed_techs:
                    raise ValueError(
                        f"Krytyczny błąd: Stacja {station_idx} nie ma dozwolonych technologii w RandomSolver.")
                current_assignment.append(random.choice(station_specific_allowed_techs))

            current_score = self.synergy_calculator.calculate_synergy_score(current_assignment)
            if current_score > best_score:
                best_score = current_score
                best_assignment = list(current_assignment)

            if num_trials >= 10 and (i + 1) % (num_trials // 10) == 0:
                print(f"  RandomSolver: Próba {i + 1}/{num_trials}, Najlepszy dotychczasowy wynik: {best_score:.2f}")

        initial_assignment_from_problem = self.problem.get_initial_candidate_assignment(strategy="first")
        if initial_assignment_from_problem:
            try:
                initial_score = self.synergy_calculator.calculate_synergy_score(initial_assignment_from_problem)
                print(f"  RandomSolver: Wynik dla konfiguracji 'pierwsza dozwolona': {initial_score:.2f}")
                if initial_score > best_score:
                    print("  RandomSolver: Konfiguracja 'pierwsza dozwolona' jest lepsza niż losowe próby.")
                    best_score = initial_score
                    best_assignment = initial_assignment_from_problem
            except Exception as e:
                print(f"  RandomSolver: Nie można było ocenić konfiguracji 'pierwsza dozwolona': {e}")

        print(f"RandomSolver zakończony. Najlepszy znaleziony wynik: {best_score:.2f}")
        return best_assignment, best_score


class GreedySolver:
    """
    Zoptymalizowana wersja GreedySolvera, która używa obliczeń przyrostowych (delta),
    dzięki czemu działa znacznie szybciej.
    """

    def __init__(self, qap_problem, synergy_map):
        self.problem = qap_problem
        self.calculator = QAPSolver(qap_problem, synergy_map)

    def solve(self, initial_assignment, max_iterations=100):
        if not initial_assignment or self.problem.num_stations == 0:
            return None, -float('inf')

        current_assignment = list(initial_assignment)
        current_score = self.calculator.calculate_synergy_score(current_assignment)
        print(f"  GreedySolver: Wynik początkowy: {current_score:.2f}")

        for iteration in range(max_iterations):
            best_move = None
            best_score_gain = 0.0

            for station_idx in range(self.problem.num_stations):
                original_tech = current_assignment[station_idx]
                allowed_techs = self.problem.get_allowed_technologies_for_station(station_idx)

                if len(allowed_techs) <= 1:
                    continue

                for new_tech in allowed_techs:
                    if new_tech == original_tech:
                        continue

                    score_delta = self.calculator.calculate_score_delta(
                        current_assignment, station_idx, new_tech
                    )

                    if score_delta > best_score_gain:
                        best_score_gain = score_delta
                        best_move = (station_idx, new_tech)

            if best_move:
                change_idx, new_tech_for_change = best_move
                current_assignment[change_idx] = new_tech_for_change
                current_score += best_score_gain

                print(
                    f"  GreedySolver: Iteracja {iteration + 1}, Nowy wynik: {current_score:.2f} (Poprawa o {best_score_gain:.2f})")
            else:
                print(f"  GreedySolver: Osiągnięto lokalne optimum po {iteration} iteracjach.")
                break

        if iteration == max_iterations - 1 and best_move:
            print(f"  GreedySolver: Osiągnięto maksymalną liczbę iteracji ({max_iterations}).")

        return current_assignment, current_score


# ==================== ALGORYTMY GENETYCZNE ====================

class BaseGeneticSolver:
    """
    Bazowa klasa dla algorytmów genetycznych.
    Zawiera wspólną funkcjonalność dla Simple GA i Dedicated GA.
    """

    def __init__(self, qap_problem, synergy_map, proximity_function=None):
        self.problem = qap_problem
        self.calculator = QAPSolver(qap_problem, synergy_map, proximity_function)
        self.synergy_map = synergy_map

        # Parametry algorytmu genetycznego
        self.population_size = 200
        self.max_evaluations = 1000  # Maksymalna liczba ocen
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.tournament_size = 2
        self.elitism_size = 1

    def _create_random_individual(self) -> List[str]:
        """Tworzy losowego osobnika (przypisanie technologii do stacji)."""
        individual = []
        for station_idx in range(self.problem.num_stations):
            allowed_techs = self.problem.get_allowed_technologies_for_station(station_idx)
            individual.append(random.choice(allowed_techs))
        return individual

    def _evaluate_individual(self, individual: List[str]) -> float:
        """Ocenia osobnika (oblicza wynik synergii)."""
        return self.calculator.calculate_synergy_score(individual)

    def _tournament_selection(self, population: List[List[str]], scores: List[float]) -> List[str]:
        """Selekcja turniejowa - wybiera najlepszego z losowej grupy."""
        tournament_indices = random.sample(range(len(population)), min(self.tournament_size, len(population)))
        best_idx = max(tournament_indices, key=lambda i: scores[i])
        return population[best_idx].copy()

    def _uniform_crossover(self, parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
        """
        Uniform crossover - dla każdej pozycji losowo wybiera rodzica.
        Uwzględnia ograniczenia dozwolonych technologii dla każdej stacji.
        """
        child1, child2 = [], []

        for station_idx in range(len(parent1)):
            allowed_techs = self.problem.get_allowed_technologies_for_station(station_idx)

            # Sprawdź, które technologie z rodziców są dozwolone
            p1_tech = parent1[station_idx] if parent1[station_idx] in allowed_techs else None
            p2_tech = parent2[station_idx] if parent2[station_idx] in allowed_techs else None

            if random.random() < 0.5:
                # Wybierz z parent1
                child1.append(p1_tech if p1_tech else random.choice(allowed_techs))
                child2.append(p2_tech if p2_tech else random.choice(allowed_techs))
            else:
                # Wybierz z parent2
                child1.append(p2_tech if p2_tech else random.choice(allowed_techs))
                child2.append(p1_tech if p1_tech else random.choice(allowed_techs))

        return child1, child2

    def _one_point_crossover(self, parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
        """
        One-point crossover - wybiera punkt cięcia i wymienia segmenty.
        Uwzględnia ograniczenia dozwolonych technologii.
        """
        if len(parent1) <= 1:
            return parent1.copy(), parent2.copy()

        crossover_point = random.randint(1, len(parent1) - 1)

        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        # Napraw dzieci - upewnij się, że wszystkie technologie są dozwolone
        child1 = self._repair_individual(child1)
        child2 = self._repair_individual(child2)

        return child1, child2

    def _repair_individual(self, individual: List[str]) -> List[str]:
        """Naprawia osobnika - zamienia niedozwolone technologie na dozwolone."""
        repaired = []
        for station_idx, tech in enumerate(individual):
            allowed_techs = self.problem.get_allowed_technologies_for_station(station_idx)
            if tech in allowed_techs:
                repaired.append(tech)
            else:
                repaired.append(random.choice(allowed_techs))
        return repaired

    def _crossover(self, parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
        """Wykonuje crossover - równe prawdopodobieństwo uniform i one-point."""
        if random.random() < 0.5:
            return self._uniform_crossover(parent1, parent2)
        else:
            return self._one_point_crossover(parent1, parent2)

    def _mutate(self, individual: List[str]) -> List[str]:
        """Bazowa mutacja - do nadpisania w klasach dziedziczących."""
        raise NotImplementedError("Metoda _mutate musi być zaimplementowana w klasie dziedziczącej")

    def solve(self, verbose=True) -> Tuple[List[str], float]:
        if self.problem.num_stations == 0:
            print("Nie można uruchomić algorytmu genetycznego - brak stacji w problemie.")
            return None, -float('inf')

        population = [self._create_random_individual() for _ in range(self.population_size)]
        best_individual, best_score = None, -float('inf')

        if verbose:
            print(f"\nUruchamianie {self.__class__.__name__}...")
            print(f"Parametry: populacja={self.population_size}, max ocen={self.max_evaluations}, "
                  f"mutacja={self.mutation_rate}, crossover={self.crossover_rate}")

        evaluations = 0
        # Początkowa ocena populacji
        scores = []
        for ind in population:
            scores.append(self._evaluate_individual(ind))
            evaluations += 1
        gen = 0

        while evaluations < self.max_evaluations:
            gen += 1
            # Śledzenie najlepszego
            for i, score in enumerate(scores):
                if score > best_score:
                    best_score = score
                    best_individual = population[i].copy()

            if verbose and evaluations % max(1, self.max_evaluations // 10) < self.population_size:
                avg_score = sum(scores) / len(scores)
                print(f"  Po {evaluations} ocenach: Najlepszy={best_score:.2f}, Średni={avg_score:.2f}")

            new_population = []
            new_scores = []
            # Elityzm
            elite_indices = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)[:self.elitism_size]
            for idx in elite_indices:
                # new_population.append(population[idx].copy())
                elite = population[idx].copy()
                new_population.append(elite)
                new_scores.append(scores[idx])

            # Tworzenie potomków
            while len(new_population) < self.population_size and evaluations < self.max_evaluations:
                p1 = self._tournament_selection(population, scores)
                p2 = self._tournament_selection(population, scores)
                if random.random() < self.crossover_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                if random.random() < self.mutation_rate:
                    c1 = self._mutate(c1)
                if random.random() < self.mutation_rate:
                    c2 = self._mutate(c2)
                for child in (c1, c2):
                    if evaluations < self.max_evaluations:
                        # new_population.append(child)
                        # # Ocena dziecka
                        # score = self._evaluate_individual(child)
                        # evaluations += 1
                        # scores.append(score)
                        # # Trzymaj wyniki pary osobno
                        score = self._evaluate_individual(child)
                        new_population.append(child)
                        new_scores.append(score)
                        evaluations += 1
            # Przytnij populację
            # population = new_population[:self.population_size]
            # scores = scores[:len(population)]
            population = new_population
            scores = new_scores

        if verbose:
            print(f"{self.__class__.__name__} zakończony po {evaluations} ocenach. Najlepszy wynik: {best_score:.2f}")

        return best_individual, best_score


class SimpleGeneticSolver(BaseGeneticSolver):
    """
    Prosty algorytm genetyczny z podstawową mutacją.
    Mutacja polega na zmianie technologii na losową inną dozwoloną dla danej stacji.
    """

    def __init__(self, qap_problem, synergy_map, proximity_function=None):
        super().__init__(qap_problem, synergy_map, proximity_function)
        # Możesz dostosować parametry specyficzne dla Simple GA
        self.mutation_rate = 0.15

    def _mutate(self, individual: List[str]) -> List[str]:
        """
        Prosta mutacja - dla każdej stacji z pewnym prawdopodobieństwem
        zmień technologię na losową inną dozwoloną.
        """
        mutated = individual.copy()

        for station_idx in range(len(mutated)):
            if random.random() < 0.1:  # Prawdopodobieństwo mutacji pojedynczego genu
                allowed_techs = self.problem.get_allowed_technologies_for_station(station_idx)
                if len(allowed_techs) > 1:
                    current_tech = mutated[station_idx]
                    # Wybierz inną technologię niż obecna
                    other_techs = [tech for tech in allowed_techs if tech != current_tech]
                    if other_techs:
                        mutated[station_idx] = random.choice(other_techs)

        return mutated


class DedicatedGeneticSolver(BaseGeneticSolver):
    """
    Dedykowany algorytm genetyczny wykorzystujący wiedzę o sąsiedztwie stacji
    i synergii między technologiami do inteligentnej mutacji.
    """

    def __init__(self, qap_problem, synergy_map, proximity_function=None):
        super().__init__(qap_problem, synergy_map, proximity_function)
        # Parametry specyficzne dla Dedicated GA
        self.mutation_rate = 0.12
        self.crossover_rate = 0.5
        self.neighborhood_threshold = 0.1  # Próg odległości dla uznania stacji za sąsiadów

        # Prekalkulacja sąsiedztw dla wydajności
        self._calculate_neighborhoods()
        self._calculate_technology_preferences()

    def _calculate_neighborhoods(self):
        """Prekalkuluje sąsiedztwa stacji na podstawie odległości."""
        self.neighborhoods = {}

        # Znajdź maksymalną odległość w problemie
        max_distance = 0
        for i in range(self.problem.num_stations):
            for j in range(i + 1, self.problem.num_stations):
                max_distance = max(max_distance, self.problem.distance_matrix_D[i][j])

        # Próg sąsiedztwa jako procent maksymalnej odległości
        distance_threshold = max_distance * self.neighborhood_threshold

        for i in range(self.problem.num_stations):
            neighbors = []
            for j in range(self.problem.num_stations):
                if i != j and self.problem.distance_matrix_D[i][j] <= distance_threshold:
                    neighbors.append(j)
            self.neighborhoods[i] = neighbors

    def _calculate_technology_preferences(self):
        """Oblicza preferencje technologii na podstawie wartości synergii."""
        self.tech_preferences = {}

        all_techs = list(self.synergy_map.keys())

        for tech in all_techs:
            # Znajdź technologie z wysoką synergią z daną technologią
            synergies = []
            for other_tech in all_techs:
                if tech != other_tech:
                    synergy = self.calculator._get_synergy(tech, other_tech)
                    synergies.append((other_tech, synergy))

            # Sortuj po synergii i weź top technologie
            synergies.sort(key=lambda x: x[1], reverse=True)
            preferred_techs = [t[0] for t in synergies[:len(synergies) // 2] if t[1] > 0]
            self.tech_preferences[tech] = preferred_techs

    def _get_neighborhood_context(self, individual: List[str], station_idx: int) -> Dict[str, int]:
        """Zwraca kontekst sąsiedztwa - jakie technologie są w pobliżu."""
        context = {}
        neighbors = self.neighborhoods.get(station_idx, [])

        for neighbor_idx in neighbors:
            neighbor_tech = individual[neighbor_idx]
            context[neighbor_tech] = context.get(neighbor_tech, 0) + 1

        return context

    def _intelligent_tech_selection(self, station_idx: int, individual: List[str]) -> str:
        """
        Inteligentny wybór technologii na podstawie sąsiedztwa i synergii.
        """
        allowed_techs = self.problem.get_allowed_technologies_for_station(station_idx)
        if len(allowed_techs) <= 1:
            return allowed_techs[0]

        # Pobierz kontekst sąsiedztwa
        neighborhood_context = self._get_neighborhood_context(individual, station_idx)

        # Ocena każdej dozwolonej technologii
        tech_scores = {}
        for tech in allowed_techs:
            score = 0.0

            # Dodaj punkty za synergię z technologiami w sąsiedztwie
            for neighbor_tech, count in neighborhood_context.items():
                synergy = self.calculator._get_synergy(tech, neighbor_tech)
                score += synergy * count

            # Dodaj bonus za technologie preferowane
            preferred_techs = self.tech_preferences.get(tech, [])
            for neighbor_tech in neighborhood_context:
                if neighbor_tech in preferred_techs:
                    score += 0.5  # Bonus za preferowaną technologię w sąsiedztwie

            tech_scores[tech] = score

        # Wybierz technologię z najwyższym wynikiem (z elementem losowości)
        if tech_scores:
            # Sortuj technologie po wyniku
            sorted_techs = sorted(tech_scores.items(), key=lambda x: x[1], reverse=True)

            # Wybierz z top 3 technologii (jeśli są dostępne) z wagami
            top_techs = sorted_techs[:min(3, len(sorted_techs))]

            if len(top_techs) == 1:
                return top_techs[0][0]

            # Wagi: najlepsza ma 50% szansy, druga 30%, trzecia 20%
            weights = [0.5, 0.3, 0.2][:len(top_techs)]
            weights = [w / sum(weights) for w in weights]  # Normalizacja

            return random.choices([tech for tech, _ in top_techs], weights=weights)[0]

        return random.choice(allowed_techs)

    def _mutate(self, individual: List[str]) -> List[str]:
        """
        Inteligentna mutacja wykorzystująca wiedzę o sąsiedztwie i synergii.
        """
        mutated = individual.copy()

        for station_idx in range(len(mutated)):
            if random.random() < 0.08:  # Niższe prawdopodobieństwo mutacji pojedynczego genu
                allowed_techs = self.problem.get_allowed_technologies_for_station(station_idx)
                if len(allowed_techs) > 1:
                    # Użyj inteligentnego wyboru technologii
                    new_tech = self._intelligent_tech_selection(station_idx, mutated)
                    if new_tech != mutated[station_idx]:
                        mutated[station_idx] = new_tech

        return mutated

if __name__ == "__main__":
    # KROK 1: Konfiguracja
    synergy_map_high_contrast_path = 'SM1_duze_roznice.csv'
    synergy_map_low_contrast_path = 'SM2_male_roznice.csv'
    synergy_map_to_use = synergy_map_low_contrast_path
    instance_to_run = 1

    # KROK 2: Wczytywanie Danych
    print("--- Wczytywanie Mapy Synergii ---")
    synergy_map = load_synergy_map_from_csv(synergy_map_to_use)

    if synergy_map:
        instance_path = os.path.join("..", "data", f"woj{instance_to_run}", "technologie.csv")
        print(f"\n--- Przetwarzanie Instancji: {instance_path} ---")

        if not os.path.exists(instance_path):
            print(f"BŁĄD: Plik instancji '{instance_path}' nie istnieje.")
        else:
            qap_problem = QAPProblem(csv_filepath=instance_path, loaded_synergy_map=synergy_map)

            if qap_problem.num_stations > 0:
                print(f"Wczytano {qap_problem.num_stations} stacji dla instancji woj{instance_to_run}.")

                # Inicjalizacja i ocena początkowego rozwiązania
                initial_assignment = qap_problem.get_initial_candidate_assignment("random")
                evaluator = QAPSolver(qap_problem, synergy_map)
                initial_score = evaluator.calculate_synergy_score(initial_assignment)
                print(f"\nWynik dla losowego przypisania początkowego: {initial_score:.2f}")

                # RandomSolver
                print("\nUruchamianie RandomSolver...")
                random_solver = RandomSolver(qap_problem, synergy_map)
                rs_assignment, rs_score = random_solver.solve(num_trials=100)
                print(f"RandomSolver - Najlepszy wynik: {rs_score:.2f}")

                # GreedySolver (zaczynając od najlepszego z losowych)
                print("\nUruchamianie GreedySolver...")
                greedy_solver = GreedySolver(qap_problem, synergy_map)
                start_assignment_for_greedy = rs_assignment if rs_assignment else initial_assignment
                gs_assignment, gs_score = greedy_solver.solve(initial_assignment=start_assignment_for_greedy)
                print(f"GreedySolver - Najlepszy wynik: {gs_score:.2f}")

                # Simple GA
                print("\nUruchamianie SimpleGeneticSolver...")
                simple_ga = SimpleGeneticSolver(qap_problem, synergy_map)
                simple_ga.max_evaluations = 1000
                sg_assignment, sg_score = simple_ga.solve()
                print(f"SimpleGeneticSolver - Najlepszy wynik: {sg_score:.2f}")

                # Dedykowany GA
                print("\nUruchamianie DedicatedGeneticSolver...")
                dedicated_ga = DedicatedGeneticSolver(qap_problem, synergy_map)
                dedicated_ga.max_evaluations = 1000
                dg_assignment, dg_score = dedicated_ga.solve()
                print(f"DedicatedGeneticSolver - Najlepszy wynik: {dg_score:.2f}")

                # Podsumowanie
                print("\n--- PODSUMOWANIE DLA INSTANCJI ---")
                print(f"Instancja: woj{instance_to_run}")
                print(f"Mapa Synergii: {synergy_map_to_use}")
                print(f"Wynik początkowy (losowy): {initial_score:.2f}")
                print(f"Wynik RandomSolver (5000 prób): {rs_score:.2f}")
                print(f"Wynik GreedySolver: {gs_score:.2f}")
                print(f"Najlepsze znalezione przypisanie (Greedy): {gs_assignment}")
                print(f"Wynik Simple GA: {sg_score:.2f}")
                print(f"Wynik Dedicated GA: {dg_score:.2f}")
                print(f"Najlepsze przypisanie (Dedykowany): {dg_assignment}")
