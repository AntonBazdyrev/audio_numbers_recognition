import numpy as np

from Levenshtein import distance
from catalyst import metrics, callbacks


def convert_to_str(list_num):
    return str(int(''.join(str(x) for x in list_num)))

def cer(logits, targets):
    preds = logits.argmax(axis=1)
    scores = []
    for p, t in zip(preds, targets):
        p = convert_to_str(p.cpu().numpy())
        t = convert_to_str(t.cpu().numpy())
        scores.append(distance(t, p)/len(t))
    return np.mean(scores)

cer_metric = metrics.FunctionalBatchMetric(
    metric_fn=cer,
    metric_key="cer_metric",
)
cer_metric.reset()

cer_callback = callbacks.metric.FunctionalBatchMetricCallback(
    metric=cer_metric,
    input_key='logits',
    target_key='targets',
)
