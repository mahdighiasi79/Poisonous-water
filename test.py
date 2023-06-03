import numpy as np
import helper_functions as hf
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("Dataset2.csv")
    print(hf.ExtractValues(df["poisonous"]))
