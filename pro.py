import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


header = ['preg', 'plas', 'pres', 'skin' , 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('./data/pima-indians-diabetes.data.csv', names= header)

array = data.values


# data.plot(kind="density", subplots=True, figsize=(12,10), layout=(3,3), sharex=False, sharey=False)
# plt.savefig('./results/density.png')
# plt.show()

# data.plot(kind="box", subplots=True, figsize=(12,10), layout=(3,3), sharex=False, sharey=False)
# plt.savefig("./results/boxplot.png")
# plt.show()

# data.hist(figsize=(12,10), bins=5)
# plt.tight_layout
# plt.savefig("./results/histogram.png")

