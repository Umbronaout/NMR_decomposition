import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

seed = 42

random.seed(seed)
np.random.seed(seed)

class optimizer():
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    @staticmethod
    def Lorentzian(x: np.ndarray, amp, mean, gamma):
        return amp / (np.pi * gamma * (1 + ((x - mean) / gamma) ** 2))
    
    @staticmethod
    def Gaussian(x: np.ndarray, amp, mean, sigma):
        return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    def combined_function(self, functions:list):
        """
        Nested function that returns a combination of Gaussians
        param:
        Gaussians(list):   List of parameters (tuples) of Lorentzians/Gaussians - (type, amp, mean, gamma/sigma)
        """
        combined_result = np.zeros(self.x_data.size)
        for function_params in functions:
            if function_params[0] == 0:
                combined_result += self.Gaussian(self.x_data, *function_params[1:])
            elif function_params[0] == 1:
                combined_result += self.Lorentzian(self.x_data, *function_params[1:])
            else:
                raise ValueError(f"Expected 0 or 1 as the first element in settings to specify Gaussian or Lorentzian. Got {function_params[0]}")

        return combined_result    

    def loss_function(self, function_params:list):
        """
        Nested loss function for optimization
        params:
        function_params(list):  List with function settings
        """

        fitted_response = self.combined_function(function_params)
        residuals = self.y_data - fitted_response
        loss = np.dot(residuals, residuals)

        regularization = len(function_params) / 100

        #print(f"{loss:.8f}\t{regularization:.8f}")

        return loss + regularization

    def auto_initial_guess(self, verbose=False):
        # Amplitudes, means and local extrema
        amps = []
        means = []
        minimums = []
        maximums = []
        types = []
        for idx, point in enumerate(self.y_data[1:-1]):
            if point > self.y_data[idx] and point > self.y_data[idx + 2]:
                means.append(self.x_data[idx + 1])
                amps.append(point)
                maximums.append(idx)
                types.append(0)
            elif point < self.y_data[idx] and point < self.y_data[idx + 2]:
                minimums.append(idx)
        
        # Gammas
        gammas = []
        minimums = np.array(minimums)
        for maxima in maximums:
            if minimums.size != 0:
                closest_minima_index = (minimums - maxima).argmin()
                gammas.append((self.x_data[maxima] - self.x_data[closest_minima_index]) / 8)
            else:
                gammas.append(1/1000)
        
        init_functions = list(zip(types, amps, means, gammas))

        if verbose:
            print("Initial Functions:")
            for function in init_functions:
                print(f"{'Gaussian' if function[0] == 0 else 'Lorentzian'}\tAmp:\t{function[1]:.4f}\tMean:\t{function[2]:.4f}\tGamma:\t{function[3]:.8f}")

        return init_functions
    
    def optimize(self, max_iters:int, population_size:int, init_mutation_chance:float, display=False):
        population = []
        init_guess = self.auto_initial_guess(verbose=True) #TODO
        for _ in range(population_size):
            population.append(copy.copy(init_guess))

        def cosine_decay(iteration, cycles=np.sqrt(max_iters), decay_factor=3):
            return (np.cos(2 * np.pi * iteration / max_iters * cycles) + 1) / 2 * np.exp(1 / decay_factor * (1 / (iteration / max_iters + 1.2) + 1 / (iteration / max_iters - 1.2)))

        def mutate(population, generation_idx):
            mutation_chance = init_mutation_chance * cosine_decay(iteration=generation_idx)

            def peak_mutation(peak, duplicate=False):
                new_peak = list(copy.copy(peak))
                # Gaussian <-> Lorentzian
                if random.uniform(0, 1) <= mutation_chance:
                    new_peak[0] = 1 if peak[0] == 0 else 0
                for mutation_idx in range(1, 4):
                    if duplicate:
                        new_peak[mutation_idx] *= (1 + random.choice([1, -1]) * random.uniform(0.01, 0.5))
                    else:
                        if random.uniform(0, 1) <= mutation_chance:
                            new_peak[mutation_idx] *= (1 + random.choice([1, -1]) * random.uniform(0.01, 0.1))

                return(tuple(new_peak))
            
            new_population = []
            for individual in population:
                new_individual = [peak_mutation(peak) for peak in individual]

                # Delete random peak
                if len(individual) > 1 and random.uniform(0, 1) <= mutation_chance:
                    del new_individual[np.random.choice(list(range(len(individual))))]

                # Add random mutated copy of a peak
                elif random.uniform(0, 1) <= mutation_chance:
                    new_individual.append(peak_mutation(random.choice(new_individual), duplicate=True))

                # Add artifitial peak
                elif random.uniform(0, 1) <= mutation_chance:
                    difference_array = (self.y_data - self.combined_function(new_individual)) ** 2
                    normalization_constant = np.sum(difference_array)
                    if normalization_constant == 0:
                        continue
                    probabilities = difference_array / normalization_constant
                    choice_idx = np.random.choice(list(range(probabilities.size)), p=probabilities)
                    artificial_peak = [0, np.sqrt(difference_array[choice_idx]), self.x_data[choice_idx], 1/1000]
                    new_individual.append(artificial_peak)

                new_population.append(new_individual)

            return new_population

        def propagate(population):
            fitnesses = np.array([self.loss_function(individual) for individual in population])
            fitnesses = 1 / fitnesses
            probabilities = fitnesses / np.sum(fitnesses)

            new_population_indices = np.random.choice(len(population), size=population_size, p=probabilities)
            new_population = [population[idx] for idx in new_population_indices]
            
            return new_population

        for idx in tqdm(range(max_iters)):
            population = mutate(population, idx)
            population = propagate(population)

        fitnesses = np.array([self.loss_function(individual) for individual in population])
        best_idx = fitnesses.argmin()
        best = population[best_idx]

        if display:
            fitted_curve = self.combined_function(best)

            plt.figure(figsize=(10, 6))
            plt.plot(list(self.x_data), list(fitted_curve), label='Fitted', linestyle='--', color='crimson')
            plt.plot(list(self.x_data), list(self.y_data), label='Original', color='navy')

            for idx, peak in enumerate(best):
                if peak[0] == 0:
                    y_values = list(self.Gaussian(self.x_data, *peak[1::]))
                elif peak[0] == 1:
                    y_values = list(self.Lorentzian(self.x_data, *peak[1::]))
                plt.plot(list(self.x_data), y_values, label=f"peak_{idx + 1}", alpha=0.2, color="black")

            plt.xlabel('Chemical Shift (ppm)')
            plt.ylabel('Intensity')
            plt.title('NMR Spectrum')
            plt.legend()
            plt.gca().invert_xaxis()
            plt.show()

        return(best)

if __name__ == "__main__":
    optim = optimizer(x_data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), y_data=np.array([0, 0, 0, 1, 3, 10, 3, 1, 0, 0, 0]))
    print(optim.optimize(population_size=1000, max_iters=500, init_mutation_chance=0.2, display=True))