# INSTALL PACKAGES ---------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# MAKE A FUNCTION FOR THE DESCRIPTIONS -------------------------------------------------------------
club_info = pd.read_csv("./data/lending_club_info.csv")

def feat_info(col_name):
    return club_info[club_info["LoanStatNew"]==col_name]["Description"][0]

feat_info("loan_amnt")