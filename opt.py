import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

data=pd.read_csv('opt.csv')
df=pd.DataFrame(data)
x=np.array(df['Specific_coll_area'])
y=np.array(df['coll_eff'])

def model2(x,a0,a1,a2):
    num=np.exp(-a0*x)
    den=a1+a2*x
    return 100*(1-num/den)

param_opt2,param_cov2=optimize.curve_fit(model2,x,y)
aopt0=param_opt2[0]
aopt1=param_opt2[1]
aopt2=param_opt2[2]


model_y=model2(x,aopt0,aopt1,aopt2)

fig, axis = plt.subplots()
axis.plot(x, y, linestyle=" ", marker="o", color="black", label="Measured")
axis.plot(x, model_y, linestyle="-", marker=None, color="red", label="Modeled")
axis.set_xlabel("Specific collection area")
axis.set_ylabel("Collection efficiency")
axis.grid(True)
axis.legend(loc="best")

plt.show()

