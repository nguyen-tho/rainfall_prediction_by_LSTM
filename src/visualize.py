from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

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
    
def show_training_log(history):
    #hist is training history
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    fig,ax=plt.subplots(1,2,figsize=(16,8))
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[0].set_title('Training Loss')
    ax[1].set_title('Training Accuracy')
    ax[0].plot(hist['epoch'], hist['loss'], label='train_loss')
    ax[0].plot(hist['epoch'], hist['val_loss'], label='val_loss')
    plt.plot(hist['epoch'], hist['accuracy'], label='train_acc')
    plt.plot(hist['epoch'], hist['val_accuracy'], label='val_acc')
    plt.legend()

    plt.show()
    
    
def show_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    class_labels = ['Không mưa', 'Mưa không đáng kể', 'Mưa nhỏ', 'Mưa', 'Mưa vừa', 'Mưa to', 'Mưa rất to']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.title("Confusion Matrix for test set 1")
    plt.show()
    
def print_report(y_true, y_pred):
    #y_true is true label, y_pred is predicted label
    print(classification_report(y_true, y_pred))
    
    
    
