
# coding: utf-8

# In[ ]:


from sklearn.metrics import homogeneity_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, zero_one_loss, classification_report
import pandas as pd
import pickle
model_filename = 'model.sav'
pred_filename = 'pred.sav'


def unsupervised_results(predicted_labels_unsupervised,recieved_labels):
    print(homogeneity_score(predicted_labels_unsupervised,recieved_labels))


def accuracy_check(model, x_test, y_test):
#     Load here the model and validation set data to test
    # test accuracy
    y_pred_ts = model.predict(x_test)
    predictions_ts = [round(value) for value in y_pred_ts]
    accuracy2 = accuracy_score(y_test, predictions_ts)
    print("Accuracy on test: %.2f%%" % (accuracy2 * 100.0))
    baccuracy2 = balanced_accuracy_score(y_test, predictions_ts)
    print("Balanced Accuracy on test: %.2f%%" % (baccuracy2 * 100.0))
    z12 = zero_one_loss(y_test, predictions_ts)
    print("Zero One Loss on test: %.2f%%" % (z12 * 100))
    class_rep2 = classification_report(y_test, predictions_ts)
    print(class_rep2)

df = pd.read_csv('test_data.csv', index_col=0)
target = df['target']
del df['target']
loaded_model = pickle.load(open(model_filename, 'rb'))
loaded_result = pickle.load(open(pred_filename, 'rb'))

unsupervised_results(target, loaded_result)
accuracy_check(loaded_model, df, target)
