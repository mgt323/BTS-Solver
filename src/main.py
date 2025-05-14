import csv
import math
import random
import copy

DEFAULT_SYNERGY_MAP = {
    "GSM": {"GSM": 2, "UMTS": 4, "LTE": 2, "5G": 1, "IOT": 1},
    "UMTS": {"GSM": 4, "UMTS": 3, "LTE": 7, "5G": 3, "IOT": 2},
    "LTE": {"GSM": 2, "UMTS": 7, "LTE": 5, "5G": 10, "IOT": 3},
    "5G": {"GSM": 1, "UMTS": 3, "LTE": 10, "5G": 8, "IOT": 4},
    "IOT": {"GSM": 1, "UMTS": 2, "LTE": 3, "5G": 4, "IOT": 2},
}

ALL_AVAILABLE_TECHNOLOGIES = sorted(list(DEFAULT_SYNERGY_MAP.keys()))


class QAPProblem:
    """
    Reprezentuje problem QAP, wczytuje dane o stacjach,
    oblicza macierz odległości.
    """

    def __init__(self, csv_filepath):
        self.csv_filepath = csv_filepath
        self.stations = []
        self.coordinates = []
        self.technologies = []
        self.num_stations = 0
        self.distance_matrix_D = []

        self._load_stations()
        if self.num_stations > 0:
            self._calculate_distance_matrix()

    def _load_stations(self):
        try:
            with open(self.csv_filepath, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for idx, row in enumerate(reader):  # Dodano idx dla ID stacji
                    try:
                        x = float(row['x'])
                        y = float(row['y'])
                        tech = row['technologia'].strip().upper()

                        self.stations.append({'id': idx, 'x': x, 'y': y, 'technologia': tech})
                        self.coordinates.append((x, y))
                        self.technologies.append(tech)
                        self.num_stations += 1
                    except ValueError:
                        print(f"Ostrzeżenie: Pomijanie wiersza z powodu błędu konwersji danych: {row}")
                    except KeyError:
                        print(
                            f"Ostrzeżenie: Pomijanie wiersza. Brak oczekiwanych kolumn ('x', 'y', 'technologia') w {row}")
                        return
        except FileNotFoundError:
            print(f"Błąd: Plik {self.csv_filepath} nie został znaleziony.")
            self.num_stations = 0
        except Exception as e:
            print(f"Wystąpił nieoczekiwany błąd podczas wczytywania pliku: {e}")
            self.num_stations = 0

        if self.num_stations > 0:
            print(f"Pomyślnie wczytano {self.num_stations} stacji.")
        else:
            print("Nie wczytano żadnych stacji.")

    def _calculate_distance_matrix(self):
        self.distance_matrix_D = [[0.0] * self.num_stations for _ in range(self.num_stations)]
        for i in range(self.num_stations):
            for j in range(i + 1, self.num_stations):
                coord1 = self.coordinates[i]
                coord2 = self.coordinates[j]
                dist = math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
                self.distance_matrix_D[i][j] = dist
                self.distance_matrix_D[j][i] = dist
        # print("Macierz odległości D została obliczona.") # Mniej gadatliwe

    def get_initial_technologies(self):
        return list(self.technologies)  # Zwraca kopię

    def get_unique_technologies_from_load(self):
        return sorted(list(set(self.technologies))) if self.technologies else []


class QAPSolver:  # Ta klasa jest teraz bardziej kalkulatorem funkcji celu
    """
    Klasa do obliczania wartości funkcji celu dla problemu QAP
    zorientowanego na maksymalizację synergii.
    """

    def __init__(self, qap_problem, synergy_map, proximity_function=None):
        self.problem = qap_problem
        self.synergy_map = synergy_map

        if proximity_function is None:
            self.proximity_function = lambda d: 1.0 / (1.0 + d) if d >= 0 else 0
        else:
            self.proximity_function = proximity_function

    def _get_synergy(self, tech1_name, tech2_name):
        tech1_map = self.synergy_map.get(tech1_name)
        if tech1_map:
            return tech1_map.get(tech2_name, 0)
        # print(f"Ostrzeżenie: Brak definicji synergii dla technologii '{tech1_name}' lub pary ('{tech1_name}', '{tech2_name}')")
        return 0

    def calculate_synergy_score(self, assignment_of_technologies):
        if self.problem.num_stations == 0:
            # print("Nie można obliczyć wyniku synergii - brak stacji w problemie.")
            return 0.0

        if len(assignment_of_technologies) != self.problem.num_stations:
            raise ValueError("Liczba technologii w przypisaniu musi odpowiadać liczbie stacji.")

        total_synergy_score = 0.0
        for i in range(self.problem.num_stations):
            for j in range(i + 1, self.problem.num_stations):  # Sumujemy tylko dla i < j i mnożymy przez 2
                tech_i = assignment_of_technologies[i]
                tech_j = assignment_of_technologies[j]

                synergy_value = self._get_synergy(tech_i, tech_j)
                if synergy_value == 0 and tech_i != tech_j:  # Opcjonalne ostrzeżenie, jeśli para nie ma synergii
                    pass  # print(f"Uwaga: Brak zdefiniowanej synergii dla ({tech_i}, {tech_j})")

                distance = self.problem.distance_matrix_D[i][j]
                proximity = self.proximity_function(distance)
                total_synergy_score += synergy_value * proximity

        return total_synergy_score * 2.0  # Mnożymy przez 2, bo liczyliśmy połowę par


class RandomSolver:
    """
    Solver losowy: generuje wiele losowych przypisań technologii i wybiera najlepsze.
    """

    def __init__(self, qap_problem, synergy_map, proximity_function=None, available_technologies=None):
        self.problem = qap_problem
        # QAPSolver używany jako kalkulator funkcji celu
        self.synergy_calculator = QAPSolver(qap_problem, synergy_map, proximity_function)

        self.available_technologies = available_technologies if available_technologies else ALL_AVAILABLE_TECHNOLOGIES
        if not self.available_technologies:
            raise ValueError("Lista dostępnych technologii nie może być pusta dla RandomSolver.")
        # Weryfikacja czy dostępne technologie są w mapie synergii
        for tech in self.available_technologies:
            if tech not in self.synergy_calculator.synergy_map:
                print(
                    f"Ostrzeżenie (RandomSolver): Technologia '{tech}' z 'available_technologies' nie jest kluczem w 'synergy_map'.")

    def solve(self, num_trials=1000):
        if self.problem.num_stations == 0:
            print("Nie można uruchomić RandomSolver - brak stacji w problemie.")
            return None, -float('inf')

        best_assignment = None
        best_score = -float('inf')

        print(f"\nUruchamianie RandomSolver (liczba prób: {num_trials})...")
        for i in range(num_trials):
            current_assignment = [random.choice(self.available_technologies) for _ in range(self.problem.num_stations)]
            current_score = self.synergy_calculator.calculate_synergy_score(current_assignment)

            if current_score > best_score:
                best_score = current_score
                best_assignment = list(current_assignment)

            if num_trials >= 10 and (i + 1) % (num_trials // 10) == 0:
                print(f"  RandomSolver: Próba {i + 1}/{num_trials}, Najlepszy dotychczasowy wynik: {best_score:.2f}")

        # Sprawdź również oryginalne przypisanie z pliku, jeśli jest poprawne
        initial_assignment_from_problem = self.problem.get_initial_technologies()
        if initial_assignment_from_problem:
            try:
                initial_score = self.synergy_calculator.calculate_synergy_score(initial_assignment_from_problem)
                print(f"  RandomSolver: Wynik dla oryginalnej konfiguracji z pliku: {initial_score:.2f}")
                if initial_score > best_score:
                    print("  RandomSolver: Oryginalna konfiguracja z pliku jest lepsza niż losowe próby.")
                    best_score = initial_score
                    best_assignment = initial_assignment_from_problem
            except Exception as e:
                print(f"  RandomSolver: Nie można było ocenić oryginalnej konfiguracji: {e}")

        print(f"RandomSolver zakończony. Najlepszy znaleziony wynik: {best_score:.2f}")
        return best_assignment, best_score


class GreedySolver:
    """
    Solver zachłanny (iteracyjna poprawa): zaczyna od pewnego przypisania i iteracyjnie
    dokonuje najlepszej możliwej pojedynczej zmiany, dopóki nie osiągnie lokalnego optimum.
    """

    def __init__(self, qap_problem, synergy_map, proximity_function=None, available_technologies=None):
        self.problem = qap_problem
        self.synergy_calculator = QAPSolver(qap_problem, synergy_map, proximity_function)
        self.available_technologies = available_technologies if available_technologies else ALL_AVAILABLE_TECHNOLOGIES
        if not self.available_technologies:
            raise ValueError("Lista dostępnych technologii nie może być pusta dla GreedySolver.")
        for tech in self.available_technologies:
            if tech not in self.synergy_calculator.synergy_map:
                print(
                    f"Ostrzeżenie (GreedySolver): Technologia '{tech}' z 'available_technologies' nie jest kluczem w 'synergy_map'.")

    def solve(self, initial_assignment_override=None, max_iterations=100):
        if self.problem.num_stations == 0:
            print("Nie można uruchomić GreedySolver - brak stacji w problemie.")
            return None, -float('inf')

        if initial_assignment_override:
            current_assignment = list(initial_assignment_override)
            if len(current_assignment) != self.problem.num_stations:
                raise ValueError("Długość initial_assignment_override musi odpowiadać liczbie stacji.")
        else:
            current_assignment = self.problem.get_initial_technologies()
            if not current_assignment:  # Jeśli plik był pusty lub nie wczytano technologii
                print("GreedySolver: Brak początkowego przypisania z problemu, generowanie losowego.")
                current_assignment = [random.choice(self.available_technologies) for _ in
                                      range(self.problem.num_stations)]

        # Walidacja i ewentualna korekta technologii w current_assignment
        for i in range(len(current_assignment)):
            if current_assignment[i] not in self.available_technologies:
                print(
                    f"Ostrzeżenie (GreedySolver): Technologia '{current_assignment[i]}' na stacji {i} w przypisaniu początkowym "
                    f"nie jest na liście dostępnych ({self.available_technologies}). Zastępowanie losową.")
                current_assignment[i] = random.choice(self.available_technologies)

        try:
            current_score = self.synergy_calculator.calculate_synergy_score(current_assignment)
        except Exception as e:
            print(f"Błąd przy obliczaniu początkowego wyniku dla GreedySolver: {e}")
            # Jeśli nie można obliczyć wyniku dla początkowego, spróbuj z losowym
            current_assignment = [random.choice(self.available_technologies) for _ in range(self.problem.num_stations)]
            current_score = self.synergy_calculator.calculate_synergy_score(current_assignment)

        print(f"\nUruchamianie GreedySolver...")
        print(f"  GreedySolver: Początkowe przypisanie: {current_assignment}, Wynik: {current_score:.2f}")

        for iteration in range(max_iterations):
            best_next_assignment_in_step = list(current_assignment)  # Użyj copy.deepcopy jeśli technologie to obiekty
            best_next_score_in_step = current_score
            improvement_found_this_iteration = False

            # Przejdź przez każdą stację
            for station_idx in range(self.problem.num_stations):
                original_tech_at_station = current_assignment[station_idx]

                # Spróbuj zmienić technologię na tej stacji na każdą inną dostępną
                for new_tech_candidate in self.available_technologies:
                    if new_tech_candidate == original_tech_at_station:
                        continue

                    temp_assignment = list(current_assignment)  # Kopia
                    temp_assignment[station_idx] = new_tech_candidate

                    try:
                        temp_score = self.synergy_calculator.calculate_synergy_score(temp_assignment)
                    except Exception as e:
                        # print(f"Błąd przy obliczaniu wyniku dla temp_assignment {temp_assignment}: {e}")
                        continue  # Pomiń tego sąsiada jeśli jest problem z oceną

                    # Szukamy najlepszej *możliwej* zmiany w tej iteracji
                    if temp_score > best_next_score_in_step:
                        best_next_score_in_step = temp_score
                        best_next_assignment_in_step = list(temp_assignment)  # Kopia
                        improvement_found_this_iteration = True

            # Jeśli znaleziono jakąkolwiek poprawę w tej iteracji i jest ona lepsza niż current_score
            if improvement_found_this_iteration and best_next_score_in_step > current_score:
                print(
                    f"  GreedySolver: Iteracja {iteration + 1}, Poprawa: {current_score:.2f} -> {best_next_score_in_step:.2f}.")
                # print(f"    Zmiana na: {best_next_assignment_in_step}") # Może być zbyt szczegółowe
                current_assignment = best_next_assignment_in_step
                current_score = best_next_score_in_step
            else:
                print(
                    f"  GreedySolver: Iteracja {iteration + 1}. Brak dalszej poprawy (najlepszy sąsiad: {best_next_score_in_step:.2f} vs obecny: {current_score:.2f}). Lokalne optimum lub brak poprawy.")
                break

        if iteration == max_iterations - 1 and improvement_found_this_iteration and best_next_score_in_step > current_score:
            print(
                f"  GreedySolver: Osiągnięto maksymalną liczbę iteracji ({max_iterations}), ostatnia poprawa do {current_score:.2f}.")

        print(f"GreedySolver zakończony. Najlepszy znaleziony wynik: {current_score:.2f}")
        return current_assignment, current_score


# --- Główna część skryptu (przykład użycia) ---
if __name__ == "__main__":
    csv_data = """x,y,technologia
10.0,20.0,LTE
12.5,21.5,5G
10.5,19.0,LTE
30.0,50.0,UMTS
32.0,51.0,GSM
50.0,50.0,IOT
11.0,22.0,GSM
15.0,15.0,5G
"""  # Dodano więcej stacji dla lepszych testów
    csv_filename = "stations_solver_example.csv"
    with open(csv_filename, "w", encoding="utf-8") as f:
        f.write(csv_data)

    print(f"Dostępne technologie dla solverów: {ALL_AVAILABLE_TECHNOLOGIES}")
    print(f"Używana mapa synergii: {DEFAULT_SYNERGY_MAP}")

    # 1. Utwórz problem QAP
    qap_problem_instance = QAPProblem(csv_filepath=csv_filename)

    if qap_problem_instance.num_stations > 0:
        initial_tech_assignment = qap_problem_instance.get_initial_technologies()
        print(f"\nPoczątkowe przypisanie technologii z pliku: {initial_tech_assignment}")

        # Użyj QAPSolver (kalkulatora) do oceny początkowej konfiguracji
        evaluator = QAPSolver(qap_problem_instance, DEFAULT_SYNERGY_MAP)
        try:
            initial_score_direct_eval = evaluator.calculate_synergy_score(initial_tech_assignment)
            print(f"Wynik synergii dla konfiguracji z pliku (bezpośrednia ocena): {initial_score_direct_eval:.2f}")
        except Exception as e:
            print(f"Błąd przy bezpośredniej ocenie konfiguracji z pliku: {e}")
            initial_score_direct_eval = -float('inf')

        # --- Test RandomSolver ---
        random_solver = RandomSolver(qap_problem_instance, DEFAULT_SYNERGY_MAP,
                                     available_technologies=ALL_AVAILABLE_TECHNOLOGIES)
        rs_assignment, rs_score = random_solver.solve(num_trials=2000)  # Zwiększono liczbę prób
        if rs_assignment:
            print(f"RandomSolver - Najlepsze przypisanie: {rs_assignment}, Wynik: {rs_score:.2f}")

        # --- Test GreedySolver ---
        # Może zacząć od konfiguracji z pliku lub od najlepszej z RandomSolver (jeśli lepsza)
        greedy_start_assignment = initial_tech_assignment
        greedy_start_score = initial_score_direct_eval

        if rs_score > greedy_start_score:
            print("\nGreedySolver rozpocznie od wyniku RandomSolver, ponieważ jest lepszy od początkowego.")
            greedy_start_assignment = rs_assignment

        if not greedy_start_assignment:  # Jeśli initial_tech_assignment było puste
            greedy_start_assignment = None  # Pozwól GreedySolver wygenerować losowe

        greedy_solver = GreedySolver(qap_problem_instance, DEFAULT_SYNERGY_MAP,
                                     available_technologies=ALL_AVAILABLE_TECHNOLOGIES)
        # Przekazanie initial_assignment_override jest opcjonalne, jeśli None, GreedySolver użyje danych z pliku.
        gs_assignment, gs_score = greedy_solver.solve(initial_assignment_override=greedy_start_assignment,
                                                      max_iterations=50)
        if gs_assignment:
            print(f"GreedySolver - Najlepsze przypisanie: {gs_assignment}, Wynik: {gs_score:.2f}")

        print("\n--- Podsumowanie ---")
        print(f"Wynik początkowy (z pliku): {initial_score_direct_eval:.2f}")
        print(f"Wynik RandomSolver: {rs_score:.2f}")
        print(f"Wynik GreedySolver: {gs_score:.2f}")

    else:
        print("Zakończono z powodu braku wczytanych stacji.")