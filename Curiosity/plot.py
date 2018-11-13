import seaborn as sns   
import matplotlib.pyplot as plt
import pandas as pd 

df= pd.read_csv('/home/ameya/Mario/A3C/save/sparse.csv')
x1=df['No. Steps']
y1= df['Average reward']
z1= df['algorithm']
#print (x1, y1, x2, y2)
ax= sns.lineplot(x=x1, y=y1, hue=z1)
plt.show()