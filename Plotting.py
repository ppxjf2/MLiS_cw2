import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy import optimize
import math


def func(x, a, b, c):
    return a * np.log(b * x+1)  + c
def func2(x, a, b, c):
    return a * math.log(b * x+1) + c
def plot_results(df):


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
    g=np.zeros(2000)
    for i in range(len(g)):
        g[i]=func2(i+1,a[0],a[1],a[2])

    p=range(4000)
    #plt.scatter(x,g)
    plt.plot(p, func(p, *a),color="black")
    print(func2(10000,a[0],a[1],a[2]))
    

    #plt.plot(polyline, model(polyline),color="black", label='3 degree polynomial model', linewidth= 0.5)
    # model-= 0.5
    # value = optimize.root(model,0.5)
    # plt.axhline(0.5, linewidth = 0.5, color = "red", linestyle = "dashed")
    # plt.axvline(value.x, linewidth= 0.5, color = "red", linestyle = "dashed")
    #plt.ylim([0,1200])
    plt.xlim([0,4000])
    plt.show()
    return 1





name= "Drone-epochs 2000 18-01-2023_15-38-38.csv"
df = pd.read_csv(f'{name}', sep=',', index_col=False)    
value= plot_results(df)