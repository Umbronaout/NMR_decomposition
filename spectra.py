import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union
import os
from optim_tools import optimizer


class spectra():
    def __init__(self, file=None):
        if file is not None:
            self.load(file)

    def load(self, file_path, normalize=True):
        with open(file_path, 'r') as file:
            nmrium_data = json.load(file)
            self.x_data = np.array(nmrium_data['data']['spectra'][0]['data']['x'])
            self.y_data = np.array(nmrium_data['data']['spectra'][0]['data']['re'])
        
        if normalize:
            self.normalization_constant = np.max(self.y_data)
            self.y_data = self.y_data / self.normalization_constant

        file_name = os.path.basename(file_path)
        self.name = os.path.splitext(file_name)[0]

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(list(self.x_data), list(self.y_data), label='NMR Spectrum')
        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Intensity')
        plt.title('NMR Spectrum')
        plt.legend()
        plt.gca().invert_xaxis()
        plt.show()

    @staticmethod
    def find_value_index(array, value):
        """
        Finds closest match to a value in an np.array and returns its index
        params
        array(np.ndarray):  array to find a value in
        value(int, float):  value to be found

        return
        index(int): index of the closest value found
        """

        if not any([isinstance(value, type) for type in [int, float]]):
            raise TypeError("value in fit_input must be an integer or a float.")
        elif not isinstance(array, np.ndarray):
            raise TypeError("array in fit_input must be a numpy array.")
        
        index = np.abs(array - value).argmin()

        return index

    def clip(self, start:Optional[Union[int, float]]=None, end:Optional[Union[int, float]]=None, adjust:bool=True):
        """
        Clips the X and Y data in a certain range
        params
        start(int, float):  X value of the clipped part
        end(int, float):    X value of the clipped part
        adjust(bool):       Find closest zero value to start/end the clip if True
        """

        if self.x_data.size == 0 or self.y_data.size == 0:
            raise LookupError("No data assigned")
        elif start == None and end == None:
            raise ValueError("start and end can not be both None")
        elif start < np.min(self.x_data) or end > np.max(self.x_data):
            raise ValueError("start or end value out of data range")

        # Find closest value to start and end in data
        if start == None:
            start_index = 0
        else:
            start_index = self.find_value_index(self.x_data, start)

        if end == None:
            end_index = 0
        else:
            end_index = self.find_value_index(self.x_data, end)

        if adjust:
            # Move start to closest non-zero value
            while self.y_data[start_index] != 0:
                start_index -= 1
            while self.y_data[start_index + 1] == 0:
                start_index += 1

            # Move end to closest non-zero value
            while self.y_data[end_index] != 0:
                end_index += 1
            while self.y_data[end_index - 1] == 0:
                end_index -= 1
        
        self.x_data = self.x_data[start_index:end_index]
        self.y_data = self.y_data[start_index:end_index]

    def decompose(self):
        """
        Splits data contained in class instance into Signle peaks
        Assumes Lorenzian shape
        """

        optim = optimizer(x_data=self.x_data, y_data=self.y_data)
        decomposed = optim.optimize(population_size=1000, max_iters=5000, init_mutation_chance=0.07, display=True)
        print(decomposed)

        return decomposed



aniline_spectra = spectra(file='aniline.nmrium')
aniline_spectra.clip(start=6.75, end=6.77)
#aniline_spectra.plot()
aniline_spectra.decompose()