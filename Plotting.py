import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy import optimize
import math


def func(x, a, b, c):
    return a * np.log(b * x+1)  + c
def func2(x, a, b, c):
    return a * math.log(b * x+1) + c
def plot_results(df,epochs):


    plt.figure(dpi=100)
    #df.plot(y='total_reward', use_index=True)
    x=range(len(df.total_reward))
    
    df["total_reward"]=df["total_reward"]/100
    a,b = optimize.curve_fit(func, x, df.total_reward,bounds=(0,[50000,50000,50000]))

    plt.scatter(x,df.total_reward)

    plt.title(f"epoch vs total cumulative reward")
    polyline = np.linspace(0, 10, 200)
    #model = np.poly1d(np.polyfit(x, df.logs, 1))
    print(a)
    print(b)
    g=np.zeros(epochs)
    for i in range(len(g)):
        g[i]=func2(i+1,a[0],a[1],a[2])

    p=range(epochs)
    #plt.scatter(x,g)
    plt.plot(p, func(p, *a),color="black")

    #estimated value after 5000 epochs
    print(func2(10000,a[0],a[1],a[2]))
    #plot heuristic value
    plt.axhline(4894.424, linewidth= 0.5, color = "red", linestyle = "dashed")

    plt.xlim([0,epochs])
    plt.show()
    return 1
    
    
    
name= "Drone-epochs 5000 18-01-2023_22-03-43.csv"
epochs= 5000

df = pd.read_csv(f'{name}', sep=',', index_col=False)    
value= plot_results(df,epochs)



#         \frac{(-3a+0.1b-c+2d^3)}{100}
# a=\left \| Target Distance \right \|
# b=abs(Drone Velocity)
# c= \left \lfloor abs(DronePitch)  \right \rfloor 
# d=1+TargetsHit