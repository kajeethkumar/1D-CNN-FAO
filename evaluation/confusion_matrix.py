import pandas as pd
from sklearn.metrics import confusion_matrix

def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm).to_csv(path, index=False)
