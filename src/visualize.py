from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def pie_plot(data_col, autopct='%1.1f%%'):
    data_values = data_col.value_counts().values
    labels = data_col.value_counts().index
    
    plt.pie(data_values, labels=labels, autopct=autopct)
    plt.show()
    
    
def scatter_3D_plot(data, group, col1, col2, col3, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    groups = data.groupby(group)
    
    for name, group in groups:
        ax.scatter(group[col1], group[col2], group[col3], label=name)
        
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_zlabel(col3)
    ax.set_title(title)
    
    ax.legend()
    
    plt.show()
    

def show_correlation_matrix(data):
    #data is correlation matrix
    plt.figure(figsize=(10,10))
    plt.title("Correlation Matrix")
    sns.heatmap(data, annot=True, cmap='coolwarm', square=True)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    plt.show()
    
    
    
    
