"""
functions to evaluate the model results
Ref: https://www.tensorflow.org/tutorials/video/video_classification#evaluate_the_model

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as mpl
import seaborn as sns


def calculate_classification_metrics(y_actual, y_pred, labels):
    """
    Calculates class-wise and averaged precision, recall, and F1-score.

    """
    cm = tf.math.confusion_matrix(y_actual, y_pred, num_classes=len(labels)).numpy()
    tp = np.diag(cm)

    precision = {}
    recall = {}
    f1_score = {}

    precision_vals = []
    recall_vals = []
    f1_vals = []

    for i in range(len(labels)):
        col = cm[:, i]
        fp = np.sum(col) - tp[i]
        row = cm[i, :]
        fn = np.sum(row) - tp[i]

        denom_precision = tp[i] + fp
        denom_recall = tp[i] + fn

        prec = tp[i] / denom_precision if denom_precision != 0 else 0.0
        rec = tp[i] / denom_recall if denom_recall != 0 else 0.0

        precision[labels[i]] = prec
        recall[labels[i]] = rec

        denom_f1 = prec + rec
        f1 = (2 * prec * rec) / denom_f1 if denom_f1 != 0 else 0.0
        f1_score[labels[i]] = f1

        precision_vals.append(prec)
        recall_vals.append(rec)
        f1_vals.append(f1)

    # averages
    precision_macro = np.mean(precision_vals)
    recall_macro = np.mean(recall_vals)
    f1_macro = np.mean(f1_vals)

    macro_avg = {
        "avg_precision": precision_macro,
        "avg_recall": recall_macro,
        "avg_f1_score": f1_macro,
    }

    return precision, recall, f1_score, macro_avg


def plot_confusion_matrix(actual, predicted, labels):
    """
    Generates confusion matrix given actual and predicted labels.

    """

    sns.set_theme(rc={'figure.figsize': (6, 6)})
    sns.set_theme(font_scale=1.1)
    cm = tf.math.confusion_matrix(actual, predicted)
    mpl.figure()

    # Create heatmap
    ax = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)

    # Labeling
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)

    mpl.tight_layout()
    mpl.show()


def get_actual_predicted_labels(model, dataset):
    """
    Function to create a list of actual ground truth values and the predictions for the model.

    """
    actual = [labels for _, labels in dataset.unbatch()]
    predicted = model.predict(dataset)

    actual = tf.stack(actual, axis=0)
    predicted = tf.concat(predicted, axis=0)
    predicted = tf.argmax(predicted, axis=1)

    return actual, predicted


def plot_graphs_from_history(history):
    """
    Function to plot loss and accuracy curves from training history.

    """

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    mpl.plot(epochs, acc, 'b', label='Training acc')
    mpl.plot(epochs, val_acc, 'r', label='Validation acc')
    mpl.title('Training and validation accuracy')
    mpl.legend()

    mpl.figure()

    mpl.plot(epochs, loss, 'b', label='Training loss')
    mpl.plot(epochs, val_loss, 'r', label='Validation loss')
    mpl.title('Training and validation loss')
    mpl.legend()

    mpl.show()
