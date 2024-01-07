import numpy as np
import matplotlib.pyplot as plt
import random

# Basic counter for np arrays
def count(l:np.array, o:any):
    return len([i for i in l if i == o])

# Some parameters
b1:float = 2.0
b2:float = 1.2
c:float  = 1.0
num_players:int = 100
num_rounds:int = 10000

# Define the new payoff matrices for the two Prisoner's Dilemmas
payoff_matrix_game1:np.matrix = np.array([[b1-c, 0.-c], [b1, 0.]])
payoff_matrix_game2:np.matrix = np.array([[b2-c, 0.-c], [b2, 0.]])

def both_cooperated(outcome):
    return np.all(outcome == 0)

def play_game(strategy, game_number):
    if game_number == 1:
        payoffs = payoff_matrix_game1[strategy]
        # Make sure probabilities are non-negative
        probabilities = (payoffs - np.min(payoffs)) + 1e-10 
        probabilities /= np.sum(probabilities)
        return np.random.choice([0, 1], p=probabilities)
    elif game_number == 2:
        payoffs = payoff_matrix_game2[strategy]
        probabilities = (payoffs - np.min(payoffs)) + 1e-10
        probabilities /= np.sum(probabilities)
        return np.random.choice([0, 1], p=probabilities)
    else:
        raise ValueError("Invalid game number")

def imitate_neighbors(player, strategies, payoffs, imitation_strength:float = 1.0):
    neighbors = [neighbor for neighbor in range(len(strategies)) if neighbor != player]
    neighbor_payoffs = [payoffs[neighbor] for neighbor in neighbors]

    # Find the most successful neighbor
    most_successful_neighbor = neighbors[np.argmax(neighbor_payoffs)]

    # Imitate the strategy of the most successful neighbor with a certain strength
    if random.random() < imitation_strength:
        return strategies[most_successful_neighbor]
    else:
        return strategies[player]

# Returns the cooperation rate for every round if back_evol is set to True
def stochastic_game(num_players:int, rounds:int, defect_prob:float = 0.01, imit_prob:float = 0.01, back_evol:bool = True):
    strategies:np.array = np.zeros(num_players, dtype=int)  # Assume everyone starts with strategy 1
    game_numbers:np.array = np.ones(num_players, dtype=int)  # Assume everyone starts with game 1
    payoffs:np.array = np.zeros(num_players)  # Track payoffs for each player
    evolution:np.array = np.zeros(rounds)   # Evolution of the game according to rounds

    for i in range(rounds):
        for player in range(num_players):
            # Check if player defects and switch game
            if random.random() < defect_prob:
                strategies[player] = 1 - strategies[player]
                game_numbers[player] = 3 - game_numbers[player]  # Switch between 1 and 2 (3-1=2 and 3-2=1 and so on ...)

            # Play the selected game
            outcome = play_game(strategies[player], game_numbers[player])

            # Update payoffs based on the outcome
            if game_numbers[player] == 1:
                payoffs[player] += payoff_matrix_game1[strategies[player]][outcome]
            else:
                payoffs[player] += payoff_matrix_game2[strategies[player]][outcome]

        # Update strategies based on the imitation of successful neighbors
        for player in range(num_players):
            strategies[player] = imitate_neighbors(player, strategies, payoffs, imit_prob)
        
        # Update game evolution
        evolution[i] = float(count(strategies, 0)) / float(num_players)
        if back_evol:
            print(f'\r{100.0*float(i)/float(rounds):9.4f}% finished...', end='')
    if back_evol:
        print('\rProcess terminated !')

    return evolution if back_evol else float(count(strategies, 0)) / float(num_players)

# Returns the cooperation rate for every round if back_evol is set to True
def simple_game(num_players:int, rounds:int, game:int, defect_prob:float = 0.01, imit_prob:float = 0.01, back_evol:bool = True):
    strategies:int = np.ones(num_players, dtype=int)  # Assume everyone starts with strategy 1
    payoffs:np.array = np.zeros(num_players)  # Track payoffs for each player
    evolution:np.array = np.zeros(rounds)   # Evolution of the game according to rounds

    for i in range(rounds):
        for player in range(num_players):
            # Check if player defects
            if random.random() < defect_prob:
                strategies[player] = 1
            elif both_cooperated(play_game(strategies[player], game)):
                strategies[player] = 0

            # Play the selected game
            outcome = play_game(strategies[player], game)

            # Update payoffs based on the outcome
            if game == 1:
                payoffs[player] += payoff_matrix_game1[strategies[player]][outcome]
            else:
                payoffs[player] += payoff_matrix_game2[strategies[player]][outcome]

        # Update strategies based on the imitation of successful neighbors
        for player in range(num_players):
            strategies[player] = imitate_neighbors(player, strategies, payoffs, imit_prob)
        
        # Update game evolution
        evolution[i] = float(count(strategies, 0)) / float(num_players)
        if back_evol:
            print(f'\r{100.0*float(i)/float(rounds):9.4f}% finished...', end='')
    if back_evol:
        print('\rProcess terminated !')

    return evolution if back_evol else float(count(strategies, 0)) / float(num_players)

# Main execution
# First, simulate for default parameters
print("Evolution defect stochastic game simulation...")
evolution_default = stochastic_game(num_players, num_rounds)
print("Evolution defect game 1 simulation...")
evolution_default_game1 = simple_game(num_players, num_rounds, 1)
print("Evolution defect game 2 simulation...")
evolution_default_game2 = simple_game(num_players, num_rounds, 2)

# Second, simulate for different values of defect_prob (rounds = 500)
defect_prob_set:list[float] = [0.001 * i for i in range(1,101)]
print("Evolution defect stochastic game simulation...")
evolution_defect:list[float]  =  [stochastic_game(num_players, 500, p, 0.01, False)    for p in defect_prob_set]
print("Evolution defect game 1 simulation...")
evolution_defect_game1:list[float] = [simple_game(num_players, 500, 1, p, 0.01, False) for p in defect_prob_set]
print("Evolution defect game 2 simulation...")
evolution_defect_game2:list[float] = [simple_game(num_players, 500, 2, p, 0.01, False) for p in defect_prob_set]

# Finally, simulate for different values of imitation probabilities (rounds = 500)
imit_prob_set:list[float] = [0.001 * i for i in range(1,101)]
print("Evolution defect stochastic game simulation...")
evolution_imit:list[float]  =  [stochastic_game(num_players, 500, 0.01, p, False)    for p in imit_prob_set]
print("Evolution defect game 1 simulation...")
evolution_imit_game1:list[float] = [simple_game(num_players, 500, 1, 0.01, p, False) for p in imit_prob_set]
print("Evolution defect game 2 simulation...")
evolution_imit_game2:list[float] = [simple_game(num_players, 500, 2, 0.01, p, False) for p in imit_prob_set]

plt.subplot(3, 1, 1)
plt.plot(range(1,num_rounds+1), evolution_default, '-')
plt.plot(range(1,num_rounds+1), evolution_default_game1, 'r-')
plt.plot(range(1,num_rounds+1), evolution_default_game2, 'g-')
plt.title("Stochastic games and simple games")
plt.xlabel("According to number of rounds")
plt.ylabel("Cooperation rate")

plt.subplot(3, 1, 2)
plt.plot(defect_prob_set, evolution_defect, '-')
plt.plot(defect_prob_set, evolution_defect_game1, 'r-')
plt.plot(defect_prob_set, evolution_defect_game2, 'g-')
plt.ylabel("Cooperation rate")
plt.xlabel("According to defection probability")

plt.subplot(3, 1, 3)
plt.plot(imit_prob_set, evolution_imit, '-')
plt.plot(imit_prob_set, evolution_imit_game1, 'r-')
plt.plot(imit_prob_set, evolution_imit_game2, 'g-')
plt.ylabel("Cooperation rate")
plt.xlabel("According to imitation probability")

plt.savefig('fig_ld_coop.pdf')

plt.show()

