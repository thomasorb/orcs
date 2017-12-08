import pandas as pd
import numpy as np
import os.path
import process

class Catalog(object):

    def __init__(self, catalog_path):

        if not isinstance(catalog_path, str): raise TypeError('catalog_path must be a string')
        self.path = catalog_path
        if os.path.exists(self.path):
            self.cat = pd.read_hdf(self.path, 'table')
        else:
            self.cat = None

    def append(self, x, y):
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        new = pd.DataFrame(np.array((x,y)).T, columns=['x', 'y'])
        if self.cat is None:
            self.cat = new
        else:
            self.cat = pd.concat([self.cat, new], ignore_index=True)

    def save(self):
        self.cat.to_hdf(self.path, 'table', mode='w')
