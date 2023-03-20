import numpy as np
import pandas as pd

deposits = pd.read_excel('MktData_CurveBootstrap_AY22-23.xls', index_col = 0, skiprows = 9, usecols = 'D:F')
deposits = deposits[:6]

print(deposits)


# we can access the dates by using the axes attribute
dates = deposits.axes[0].tolist()

print(dates)
print(dates.index)

help(usecols)

