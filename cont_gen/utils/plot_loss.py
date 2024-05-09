import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

def plot_multiple_loss(
    loss_dt: Dict[str, pd.DataFrame], 
    figsize = None,
    y_lim = None
):
    """
    Plot loss curve of multiple experiments.
    """
    fig = plt.figure(figsize=figsize)
    
    for i, (name, loss_df) in enumerate(loss_dt.items()):
        plt.plot(loss_df['step'], loss_df['loss'], color = f'C{i}', label = name)
    if y_lim is not None:
        plt.ylim((0, y_lim))
    plt.legend()