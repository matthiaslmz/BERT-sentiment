import os
from src.predict_func import get_train_loss_files, plot_loss_accuracy

def main(image_name, 
        data_path="/misc/DLshare/home/rpzqk242/BERT-sentiment/",
        save_model_eval_path = "results/",
        model_eval_image_name = "plot1.png"):

    train_loss, eval_loss, train_acc, eval_acc = get_train_loss_files(os.path.join(data_path,"out"), 'txt')
    plot_loss_accuracy(train_loss=train_loss,
                    eval_loss=eval_loss,
                    train_acc=train_acc,
                    eval_acc=eval_acc,
                    path=os.path.join(data_path, save_model_eval_path),
                    image_name = model_eval_image_name)

if __name__ == "__main__":
    main(image_name="plot1.png")