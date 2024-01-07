import random

class PlayerData:

    def __init__(self, strategy, balance):
        self._strategy = strategy
        self._balance = balance

    @property
    def strategy(self):
        return self._strategy


    def set_strategy(self, value):
        self._strategy = value

    @property
    def balance(self):
        return self._balance

    def set_balance(self, value):
        self._balance = value

    def __str__(self):
        return "Strategy: " + str(self._strategy) + " - Balance: " + str(self._balance)


r = 1.6
initial_balance = 100
cost = 1

round_count = 10
player_count = 4
current_game_state = dict()
previous_game_state = dict()


def init_game_state():

    for i in range(player_count):
        current_game_state[i] = PlayerData(random.randint(0, 1), initial_balance)

    print_game_state(current_game_state)

    contributor_count = extract_contributors(current_game_state)
    payoff = compute_payoff(contributor_count)
    update_player_balance(current_game_state, payoff)



def chose_strategy(previous_state, concerned_player):
    contributor_count = extract_contributors(previous_state)

    for player in range(player_count):
        if player != concerned_player:

            if (compute_player_payoff(True, contributor_count) > compute_player_payoff(False, contributor_count)):
                return 1
            else:
                return 0


def extract_contributors(game_state):
    count = 0

    for key in game_state.keys():
        if game_state[key].strategy == 1:
            count = count + 1

    return count



def compute_player_payoff(contribute, contributor_count):
    if (contribute):
        return ((cost * (contributor_count - 1)) * r / player_count) - cost
    else:
        return (cost * contributor_count) * r / player_count


def compute_payoff(contributor_count):
    return ((cost * contributor_count) / player_count) * r


def update_player_balance(current_game_state, payoff):
    for player in current_game_state.keys():
        if current_game_state[player].strategy == 1:
            # Probably move -cost to chose strategy
            current_game_state[player].set_balance(current_game_state[player].balance + payoff - cost)
        elif current_game_state[player].strategy == 0:
            current_game_state[player].set_balance(current_game_state[player].balance + payoff)


def print_game_state(state):
    print("--------------------------------------------------------------")
    for key in state.keys():
        print("Player:", key, " - Strategy: ", state[key].strategy, " - Balance: ", state[key].balance)


def simulate():
    for _ in range(round_count):
        previous_game_state = current_game_state

        print_game_state(current_game_state)

        for player in range(player_count):
            current_strategy = chose_strategy(previous_game_state, player) #random.randint(0, 1)
            current_game_state[player].set_strategy(current_strategy)

        contributor_count = extract_contributors(current_game_state)
        payoff = compute_payoff(contributor_count)
        update_player_balance(current_game_state, payoff)


if __name__ == "__main__":
    init_game_state();
    simulate()

