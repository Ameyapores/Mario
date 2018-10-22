import seaborn as sns   
import matplotlib.pyplot as plt
import pandas as pd 

df= pd.read_csv('/home/ameya/new_icm/save/mario_curves.csv')
x=df['No. Steps']
y= df['Total Reward']
print ()
ax= sns.lineplot(x,y)
plt.show()