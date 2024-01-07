from itertools import product
import numpy as np
from copy import deepcopy
import random
import statistics
import matplotlib.pyplot as plt


class FigureFour():
    def __init__(self):
        self.mu = 1e-3
        self.beta = 10
        self.c = 1
        self.nb_players = 4
        self.r_transitions = {
            "Immediate": [1.6, 1, 1, 1, 1], 
            "Gradual": [1.6, 1.45, 1.3, 1.15, 1], 
            "Delayed": [1.6, 1.6, 1.6, 1.6, 1], 
            "None": [1.6, 1.6, 1.6, 1.6, 1.6]}
        self.transitions = [5, 4, 3, 2, 1]
        self.list_of_strategies = None
    
    def start(self, key):
        nb_games = 100
        iterations = 100
        games, payoff, previous_moves = self.init_game(nb_games)
        print(self.play_game(key, games, payoff, previous_moves, iterations))
        return self.play_game(key, games, payoff, previous_moves, iterations)

    def play_game(self, key, games, payoff_matrix, previous_moves, iterations):
        count_cooperation = True
        total = 0
        for i in range(iterations):
            total_cooperation = 0
            for index, game in enumerate(games):
                payoffs, states, nb_cooperative_states, moves_played = self.play_public_good_games(game, previous_moves[index], key, count_cooperation)
                payoff_matrix[index] = self.update_payoff(payoffs, states, key, previous_moves[index])
                games[index] = self.update_strategy(game, moves_played, payoff_matrix[index])
                previous_moves[index] = states
                total_cooperation += nb_cooperative_states
            
            total += (total_cooperation/len(games))
            games = self.update_games(games, payoff_matrix, previous_moves, key)
            if i % 100 == 0 and i != 0:
                print((total/i)*100)
        #print(previous_moves)
        #print(payoff_matrix)
        return (total/iterations) * 100
    
    def update_strategy(self, game, moves_played, payoff_matrix):
        #print(moves_played)
        for player in range(self.nb_players):
            if payoff_matrix[player] < 0.5:
                game[player][moves_played[player]] = 1 - payoff_matrix[player] + 0.01
            else:
                game[player][moves_played[player]] = payoff_matrix[player] - 0.01
        return game
    
    def update_payoff(self, payoff_line, state, key, prev_moves):
        min_bound = self.get_min_bound(key)
        max_bound = self.get_max_max_bound(key)
        payoff_res = [0, 0, 0, 0]

        #mul = self.r_transitions[key][self.transitions[(state).count(1)]-1]
        for i in range(self.nb_players):
            value = payoff_line[i] + self.get_expect_payoff(state, key, i)
            payoff_res[i] = round((value + abs(min_bound))/(max_bound + abs(min_bound)),2)     
            #print(self.get_expect_payoff(state, key, i))
            #print(payoff_res)
            #print(prev_moves)
        return payoff_res
    
    def get_max_max_bound(self, key):
        a, b, c, d, e= [1, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 0, 0, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 0]
        list_highest = (a, b, c, d, e)
        list_max = []
        for state in list_highest:
            payoff = self.compute_pay_off(1, state, key)
            next_state = [self.transitions[state[1:].count(1)]] + state[1:]
            next_payoff = self.get_expect_payoff(next_state[1:], key, 0)
            list_max.append(payoff+next_payoff)
            #print(state, payoff, next_payoff)
        return max(list_max)
    
    def expected_number_cooperator(self, key):
        dict = {
         "Immediate": [0, 0, 0, 0, 4], 
            "Gradual": [4, 4, 1, 0, 4], 
            "Delayed": [0, 2, 2, 0, 4], 
            "None": [4, 4, 4, 4, 4]
         }
        return dict[key]
    
    
    def get_expect_payoff(self, state, key, id):
        nb_cooperators = state.count(1)
        next_transition = self.transitions[nb_cooperators] 
        new_state = [0, 0, 0, 0]
        expected_number_cooperator = self.expected_number_cooperator(key)[nb_cooperators]
        for i in range(len(new_state)):
            if i == id:
                new_state[i] = state[id]
            elif expected_number_cooperator != 0:
                new_state[i] = 1
                expected_number_cooperator -= 1
        
        next_state = [next_transition] + new_state
        payoff = self.compute_pay_off(id+1, next_state, key)
        return payoff
    
    
    def get_min_bound(self, key):
        return self.compute_pay_off(1, [5, 1, 0, 0, 0], key)
    
    def get_max_bound(self, key):
        a = self.compute_pay_off(1, [1, 1, 1, 1, 1], key) * self.r_transitions[key][4] 
        b = self.compute_pay_off(1, [1, 0, 1, 1, 1], key) * self.r_transitions[key][3]
        c = self.compute_pay_off(1, [1, 0, 0, 1, 1], key) * self.r_transitions[key][2]
        d = self.compute_pay_off(1, [1, 0, 0, 0, 1], key) * self.r_transitions[key][1]
        e = self.compute_pay_off(1, [1, 0, 0, 0, 0], key) * self.r_transitions[key][0]
        return max(a, b, c, d, e)


    
    def update_games(self, game, results, previous_moves, key):
        index_worse = min(range(len(results)), key=lambda i: statistics.mean(results[i]))
        #index_best = max(range(len(results)), key=lambda i: statistics.mean(results[i]))

        i1 = random.randint(0, len(game)-1)
        i2 = random.randint(0, len(game)-1)
        j1 = random.randint(0, 3)
        j2 = random.randint(0, 3)
        payoff_player_1 = results[i1][j1]
        payoff_player_2 = results[i2][j2]
        for i in range(self.nb_players):
            i1 = random.randint(0, len(game)-1)
            j1 = random.randint(0, 3)
            payoff_player_1 = results[i1][j1]
            payoff_player_learner = results[index_worse][i]
            p = random.random()
            if p < self.mu:
                game[index_worse][i] = self.list_of_strategies[random.randint(0, len(self.list_of_strategies)-1)]
            if p < self.compute_fitness(payoff_player_learner, payoff_player_1):
                game[index_worse][i] = game[i1][j1]
        
        """"
        liste_transposee = [list(groupe) for groupe in zip(*game)]

        for groupe in liste_transposee:
            random.shuffle(groupe)

        new_game = [list(groupe) for groupe in zip(*liste_transposee)]
        """
        return game    


    def play_public_good_games(self, players, prev_action, key, count_cooperation):
        count = 0
        payoffs = [0, 0, 0, 0]
        index_list = []
        
        nb_cooperators = (prev_action).count(1)
        state = self.transitions[nb_cooperators]
        strat = []
        for player_number, strategy in enumerate(players):
            if prev_action[player_number] == 0:
                index = nb_cooperators - 1
            else:
                index = self.nb_players + nb_cooperators - 1
            
            #print(players, strategy, index)
            action = strategy[index]
            index_list.append(index)
            proba = random.random()
            if proba <= action:
                strat.append(1)
            else:
                strat.append(0)
            
        full_state = [state] + strat
        for player in range(1, self.nb_players+1):
            payoff = self.compute_pay_off(player, full_state, key)
            payoffs[player-1] = payoff

        if count_cooperation is True:
            nb_cooperators = (full_state[1:]).count(1)
            if self.transitions[nb_cooperators] == 1:
                count += 1
        return payoffs, strat, count, index_list


    def init_game(self, nb_games):
        list_game = []
        list_payoff = []
        list_previous_move = []
        strats = self.init_strats()
        for _ in range(nb_games):
            game = []
            payoff = []
            previous_move = []
            for _ in range(self.nb_players):
                strategie = random.randint(0, len(strats))
                game.append(strats[strategie-1])
                payoff.append(0)
                previous_move.append(1)
            list_game.append(game)
            list_payoff.append(payoff)
            list_previous_move.append(previous_move)
        return list_game, list_payoff, list_previous_move
    
    def init_strats(self):
        range_values = range(2)
        strategies_list = [list(combination) for combination in product(range_values, repeat=(self.nb_players*2))]
        self.list_of_strategies = strategies_list
        return strategies_list
    
    def compute_fitness(self, learner, role_model):
        np.seterr(over='ignore')
        return np.float64((1 + np.exp(-self.beta * (role_model - learner))) ** -1)
    
    
    def compute_pay_off(self, player, state, key):
        nb_cooperators = (state[1:]).count(1)
        return (nb_cooperators * self.r_transitions[key][state[0]-1] / self.nb_players) - state[player] * self.c

figure_four = FigureFour()
print("****** NONE *****")
none = figure_four.start("None")
print("****** Delayed *****")
delayed = figure_four.start("Delayed")
print("****** Gradual *****")
gradual = figure_four.start("Gradual")
print("****** Immediate *****")
immediate = figure_four.start("Immediate")

data = [none, delayed, gradual, immediate]
# Plot the histograms with reversed axes, larger bar width, and different colors
plt.bar('None', data[0], color='yellow', width=0.5, label='None')
plt.bar('Delayed', data[1], color='red', width=0.5, label='Delayed')
plt.bar('Gradual', data[2], color='purple', width=0.5, label='Gradual')
plt.bar('Immediate', data[3], color='cyan', width=0.5, label='Immediate')


plt.title('Influence of feedback on cooperation')
plt.xlabel('')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('histo.pdf')
plt.show()
