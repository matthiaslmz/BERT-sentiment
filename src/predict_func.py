import os
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, precision_recall_fscore_support, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

def read_files_in_dir_ext(dir_route, extension):
    """
    Retrieves all files in a dir with a specific extension
    :param dir_route: (str) the path where to look files with the specific extension
    :param extension: (str) the extension to look for. E.g.: ".txt".
    :return files_ext: (list) A list of files that contains all files with said extension.
    """
    files = os.listdir(dir_route)
    files_ext = [file for file in files if file.endswith(extension)]

    return files_ext

def get_train_loss_files(output_path, extension):
    """
    Returns the latest file names of the training accuracy, training loss, validation accuracy, and 
    validation loss in the checkpoint path directory

    e.g., ['train_loss_50000.txt', 'eval_loss_50000.txt', 'train_accuracy_50000.txt', 'eval_accuracy_50000.txt']
    Parameters
    ----------
    output_path: str
        Directory where the training and validation loss and accuracy files were saved. 
        This path should be the same as `output_path` parameter in `train_model()` 
    """
    last_output = [files for files in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, files))][-1]
    last_output_path = os.path.join(output_path, last_output)
    train_loss, eval_loss, train_acc, eval_acc, _ = read_files_in_dir_ext(last_output_path, extension)
    
    return (os.path.join(last_output_path, train_loss), 
            os.path.join(last_output_path, eval_loss),
            os.path.join(last_output_path, train_acc),
            os.path.join(last_output_path, eval_acc))

def plot_loss_accuracy(train_loss, eval_loss, train_acc, eval_acc, path, image_name):
    """
    Returns the checkpoint path file name with the lowest evaluation loss to be used for evaluate_model() and
    saves the log loss + accuracy VS epoch graph.

    Parameters
    ----------
    train_loss: str
        Text file path containing the train loss

    eval_loss: str
        Text file path containing the eval loss
   
    train_acc: str
        Text file pathcontaining the train accuracy

    eval_acc: str
        Text file path containing the eval accuracy

    path: str
        Directory where the image will be saved 
    
    image_name: str
        Name of the training and validation log loss + accuracy VS epoch plot to be saved. e.g., "loss_acurracy.png"
    """

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    t_loss = pd.read_csv(train_loss, sep="\n", names=["train_loss"])
    e_loss = pd.read_csv(eval_loss, sep="\n", names=["eval_loss"])
    t_acc = pd.read_csv(train_acc, sep="\n", names=["train_acc"])
    e_acc = pd.read_csv(eval_acc, sep="\n", names=["eval_acc"])

    df_loss = pd.concat([t_loss, e_loss], axis=1)
    df_acc = pd.concat([t_acc, e_acc], axis=1)

    fig, ax = plt.subplots(figsize=(12, 10))
    min_loss_y = df_loss['eval_loss'].min()
    min_loss_x = df_loss['eval_loss'].idxmin()

    max_acc_y = df_acc['eval_acc'].max()
    max_acc_x = df_acc['eval_acc'].idxmax()

    sns.lineplot(data=df_loss, ax=ax)
    ax.set(xlabel='Checkpoints', ylabel='Loss')
    ax.legend(bbox_to_anchor=[0.5, 0.5], loc='center', labels=['train', 'eval'])
    ax.annotate("x={:.3f}, min validation loss={:.3f}".format(min_loss_x, min_loss_y), xy=(min_loss_x, min_loss_y+0.2))
    ax2 = ax.twinx()
    ax2.set(ylabel='Accuracy')
    ax2.annotate("x={:.3f}, max validation acc={:.3f}".format(max_acc_x, max_acc_y), xy=(max_acc_x-10, max_acc_y-10))

    sns.lineplot(data=df_acc, ax=ax2, legend=False)

    print("Saving image..")
    saved_path = os.path.join(path, image_name)
    plt.savefig(saved_path)
    print("Image saved at:", path)

    plt.show()
    return "checkpoint-" + str(e_loss.idxmin()[0]*1000)

    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          path=None,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Parameters
    ----------
    cm: Confusion matrix provided by sklearn

    classes: list of punctuations
    
    normalize: 'True' to normalize confusion matrix. "False" by default. 

    path: String that contains the path to save the confusion matrix image

    test_loader: A torch dataset that consist of inputs, labels, and attention tensors

    model: Model that is used for test. 

    title: title of the plot

    cmap: A matplotlib color map for the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.figure(figsize=(9,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)

    print("Saving image..")
    plt.savefig(path)
    print("Image saved at:", path)

    plt.show()