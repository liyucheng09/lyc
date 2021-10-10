from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers.trainer_callback import TrainerCallback
from lyc.utils import vector_l2_normlize
import numpy as np

metrics_computing={
    'accuracy': accuracy_score,
}

def tagging_eval_for_trainer(eval_prediction):
    """This function can be sent to huggingface.Trainer as computing_metrics funcs.

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

def pred_forward(model, eval_dl):
    all_preds = []
    all_true = []
    for batch in eval_dl:
        label = batch.pop('labels')
        outputs=model(**batch)
        all_preds.extend(outputs.logits)
        all_true.extend(label)
    
    return all_true, all_preds

def GeneralEval(model, eval_dl, writer, metrics, global_step):

    all_true, all_preds = pred_forward(model, eval_dl)
    
    results = {}
    for metric in metrics:
        results[metric] = metrics_computing[metric](all_true, all_preds)
    
    for k,v in results.items():
        writer.add_scalar(k, v, global_step)
    
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
