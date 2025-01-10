from matplotlib import pyplot as plt

def pie_plot(data_col, autopct='%1.1f%%'):
    data_values = data_col.value_counts().values
    labels = data_col.value_counts().index
    
    plt.pie(data_values, labels=labels, autopct=autopct)
    plt.show()
    
    
