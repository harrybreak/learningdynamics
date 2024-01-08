from itertools import product
import numpy as np
from copy import deepcopy
import random
import statistics

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
        self.transition = None
        self.list_of_strategies = None
        self.count = 0
    
    def start(self, key):
        nb_games = 100
        iterations = 1000
        games, payoff, previous_moves = self.init_game(nb_games, key)
        print(self.play_game(key, games, payoff, previous_moves, iterations))

    def play_game(self, key, games, payoff_matrix, previous_moves, iterations):
        count_cooperation = False
        total = 0
        for i in range(iterations):
            total_cooperation = 0
            for index, game in enumerate(games):
                payoffs, states, moves_played = self.play_public_good_games(game, previous_moves[index], key, count_cooperation) #Play the PGG
                payoff_matrix[index] = self.update_payoff(payoffs, states, key) #Transform payoff received into proba
                games[index], previous_moves[index] = self.update(game, moves_played, payoff_matrix[index], states) #Modify the strat according to previous step
                total_cooperation += self.count
                
            
            total += (total_cooperation/len(games))
            games = self.update_games(games, payoff_matrix)
            if i % 100 == 0 and i == 100: #We count only from 100 iterations to have a slow convergence
                count_cooperation = True
                print((total/i)*100)
        return (total/iterations) * 100
    
    def update_payoff(self, payoff_line, state, key):
        max_bound = self.get_max_bound(key) #Maximum possible payoff that we can obtain. We need this upper bound to make the probability
        payoff_res = [0, 0, 0, 0]

        for i in range(self.nb_players):
            value = payoff_line[i]  + self.get_expect_payoff(state, key, i) #Payoff is current payoff + expect payoff on next round
            payoff_res[i] = round((value/(max_bound)),2) #We make it into a probability in dividing by the maximum
        return payoff_res
    
    def get_max_bound(self, key):
        a, b = [1, 1, 1, 1, 1], [1, 0, 1, 1, 1] #These two matrix are the only two possible initial scenarios where we can get the maximum payoff
        list_highest = (a, b)
        list_max = []
        for state in list_highest:
            payoff = self.compute_pay_off(1, state, key)
            next_state = [self.transitions[state[1:].count(1)]] + state[1:] #Compute in which transition we will be in the next state
            next_payoff = self.get_expect_payoff(next_state[1:], key, 0)
            list_max.append(payoff+next_payoff)
        return max(list_max) #We take the highest between to return 

    
    
    def expected_number_cooperator(self, key, next_transition, state, id):
        count = 0
        next_state = [next_transition] + state
        for i in range(4):
            if i != id:
                payoff =self.compute_pay_off(i, next_state, key)
                if payoff > 0: #If the payoff is equal to zero or negative, its very low that players will cooperate so we dont take these
                    count += 1
        return count

    
    
    def get_expect_payoff(self, state, key, id):
        nb_cooperators = state.count(1)
        next_transition = self.transitions[nb_cooperators]  #We compute the next transition according to the number of cooperators
        new_state = [0, 0, 0, 0]
        expected_number_cooperator = self.expected_number_cooperator(key, next_transition, state, id) #Get the expected nb of cooperators in the state
        for i in range(len(new_state)): #Loop which establishes the state
            if i == id:
                new_state[i] = state[id]
            elif expected_number_cooperator != 0:
                new_state[i] = 1
                expected_number_cooperator -= 1
        
        next_state = [next_transition] + new_state
        payoff = self.compute_pay_off(id+1, next_state, key) 
        return payoff
    
    
    def update(self, game, moves_played, payoff_matrix, state):
        for player in range(self.nb_players):
            if payoff_matrix[player] < 0.5:
                game[player][moves_played[player]] = 1 - payoff_matrix[player] + 0.01 # Add noise asked by the paper
            else:
                game[player][moves_played[player]] = payoff_matrix[player] - 0.01 # Add noise asked by the paper
        nb_cooperators = state.count(1)
        if nb_cooperators == 4 and self.transition[nb_cooperators] == 1: # If we are in first transition we count
            self.count += 1
        return game, state

    
    def update_games(self, game, results):
        index_worse = min(range(len(results)), key=lambda i: statistics.mean(results[i])) #Take the worst possible state

        i2 = random.randint(0, len(game)-1)
        j2 = random.randint(0, 3)
        payoff_player_2 = results[i2][j2]
        for i in range(self.nb_players): #For every player in the state they try to copy
            payoff_player_1 = results[index_worse][i]
            p = random.random()
            if p < self.mu: #Random mutation
                game[index_worse][i] = self.list_of_strategies[random.randint(0, len(self.list_of_strategies)-1)]
            if p < self.compute_fitness(payoff_player_1, payoff_player_2): #Evolution by copy mecanissim
                game[index_worse][i] = game[i2][j2]
        
        return game    


    def play_public_good_games(self, players, prev_action, key, count_cooperation):
        self.count = 0
        payoffs = [0, 0, 0, 0]
        index_list = []
        
        nb_cooperators = (prev_action).count(1)
        state = self.transitions[nb_cooperators]
        strat = []
        for player_number, strategy in enumerate(players):
            if prev_action[player_number] == 0:
                index = nb_cooperators - 1 #Locate the index of our strategy in our vector according to the situation. Branch defector
            else:
                index = self.nb_players + nb_cooperators - 1 #This the branch where we cooperated previous round.
            
            action = strategy[index]
            index_list.append(index)
            proba = random.random()
            if proba <= action: #If proba we cooperate
                strat.append(1)
            else:
                strat.append(0)
            
        full_state = [state] + strat #Create the full stat
        nb_cooperators = (full_state[1:]).count(1)
        for player in range(1, self.nb_players+1):
            payoff = self.compute_pay_off(player, full_state, key)
            payoffs[player-1] = payoff

        if count_cooperation is True:  
            if self.transitions[nb_cooperators] == 1:
                self.count += 1 #Count
        return payoffs, strat, index_list


    def init_game(self, nb_games, key):
        list_game = []
        list_payoff = []
        list_previous_move = []
        strats = self.init_strats()
        self.init_transition(key)
        for _ in range(nb_games):
            game = []
            payoff = []
            previous_move = []
            for _ in range(self.nb_players): #Build every strategy, payoff, move
                strategie = random.randint(0, len(strats))
                game.append(strats[strategie-1])
                payoff.append(0)
                previous_move.append(1)
            list_game.append(game)
            list_payoff.append(payoff)
            list_previous_move.append(previous_move)
        return list_game, list_payoff, list_previous_move
    
    def init_strats(self): #Create the 256 different strategy
        range_values = range(2)
        strategies_list = [list(combination) for combination in product(range_values, repeat=(self.nb_players*2))]
        self.list_of_strategies = strategies_list
        return strategies_list
    
    def compute_fitness(self, learner, role_model): #Evolution formula
        np.seterr(over='ignore')
        return np.float64((1 + np.exp(-self.beta * (role_model - learner))) ** -1)
    
    def init_transition(self, key):
        self.transition = self.r_transitions[key]
    
    def compute_pay_off(self, player, state, key): #Payoff formula
        nb_cooperators = (state[1:]).count(1)
        return (nb_cooperators * self.r_transitions[key][state[0]-1] / self.nb_players) - state[player] * self.c

figure_four = FigureFour()
print("****** NONE *****")
figure_four.start("None")
print("****** Delayed *****")
figure_four.start("Delayed")
print("****** Gradual *****")
figure_four.start("Gradual")
print("****** Immediate *****")
figure_four.start("Immediate")
