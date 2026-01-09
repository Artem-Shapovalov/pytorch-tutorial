import csv
import os
import math

class scaler:
    def __init__(self, coeffs_csv):
        coeffs_csv_file = open(coeffs_csv, "r", newline="")
        coeffs_csv_reader = csv.reader(coeffs_csv_file)
        self.features = next(coeffs_csv_reader)
        self.mean = [float(x) for x in next(coeffs_csv_reader)]
        self.std  = [float(x) for x in next(coeffs_csv_reader)]
        coeffs_csv_file.close()

    def scale(self, feature, value):
        i = self.features.index(feature)
        # 1e-8 need to prevent division by zero
        return (value - self.mean[i]) / (self.std[i] + 1e-8)
