from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers.trainer_callback import TrainerCallback
from lyc.utils import vector_l2_normlize
import numpy as np
import torch
import pandas as pd

metrics_computing={
    'acc': accuracy_score,
}

def tagging_eval_for_trainer(eval_prediction):
    """
    Trainer专用compute_metrics函数
    This function can be sent to huggingface.Trainer as computing_metrics funcs.

    Args:
        eval_prediction ([type]): two atrributes
            - predictions
            - label_ids
    """

    predictions, labels = eval_prediction
    predictions = np.argmax(predictions, axis=-1)

    # true_predictions = [
    #     [p for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]
    # true_labels = [
    #     [l for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]

    true_predictions = [ p for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100]
    true_labels = [ l for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100]

    return {
        "accuracy_score": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions, average='micro'),
        "recall": recall_score(true_labels, true_predictions, average='micro'),
        "f1": f1_score(true_labels, true_predictions, average='micro'),
    }

def write_predict_to_file(pred_out, out_file='predictions.csv', label_list=None):
    predictions = pred_out.predictions
    labels = pred_out.label_ids

    predictions = np.argmax(predictions, axis=-1)

    if len(labels.shape) == 2:
        print('&&& Assuming tagging predictions:')
        true_predictions = [
            [(label_list[p] if label_list is not None else p) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [(label_list[l] if label_list is not None else l) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        with open(out_file, 'w', encoding='utf-8') as f:
            for p,l in zip(true_predictions, true_labels):
                f.write('pred:\t' + '\t'.join([str(i) for i in p]) + '\n')
                f.write('labels:\t' + '\t'.join([str(i) for i in l]) + '\n')
                f.write('\n')
        print(f'Save to {out_file}.')
        return

    elif len(labels.shape) == 1:
        result = {'prediction': predictions, 'labels': labels}
        df = pd.DataFrame(result)
        df.to_csv(out_file, index=False)
        print(f'Save to {out_file}.')
        return


class Evaluator:

    preprosess_func = None

    @classmethod
    def pred_forward(cls, model, eval_dl):
        model.eval()
        all_preds = []
        all_true = []
        all_loss = []
        for batch in eval_dl:
            outputs=model(**batch)
            all_preds.append(outputs.pred)
            all_true.append(outputs.target)
            all_loss.append(outputs.loss)
        
        all_true = torch.cat(all_true)
        all_preds = torch.cat(all_preds)
        all_loss = torch.stack(all_loss)
        
        return all_true.detach().numpy(), all_preds.detach().numpy(), all_loss.detach().numpy()

    @classmethod
    def GeneralEval(cls, model, eval_dl, writer=None, metrics=None, global_step=None):

        all_true, all_preds, all_loss = cls.pred_forward(model, eval_dl)
        
        results = {}

        if metrics is not None:
            for metric in metrics:
                results[metric] = metrics_computing[metric](all_true, all_preds)
        
            for k,v in results.items():
                writer.add_scalar(k, v, global_step)
        
        mean_loss = all_loss.mean()
        writer.add_scalar('Eval_loss', mean_loss, global_step)
        results['eval_loss'] = mean_loss
        
        return results

def SimCSEEvalAccComputing(preds, threshold=0.4):
    prediction=preds.prediction
    labels=pred.label_ids

    prediction = vector_l2_normlize(prediction)
    embs_a, embs_b = np.split(prediction, 2)
    sims = np.dot(embs_a, embs_b.T)
    sims = np.diag(sims)
    acc=accuracy_score(labels, sims>threshold)
    print('ACC: ', acc)
    return {'ACC' : acc}
