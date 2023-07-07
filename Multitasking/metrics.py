import numpy as np
import torch


class MetricManager:

    def __init__(self, metric_names, output_path, dataset):
        self.dataset = dataset
        self.metric_names = metric_names
        self.output_path = output_path
        self.epoch_metrics = None

        self.linked_metrics = {
            "attr_mAP": ["attr_AP_by_mask", "num_attr_by_mask", "attr_classes"],
            "attr_mAP_linear": ["attr_mAP_linear_attr_AP_by_mask", "attr_mAP_linear_num_attr_by_mask", "attr_classes"],
        }

        self.init_metrics()

    def init_metrics(self):
        self.epoch_metrics = {
            "nb_samples": list(),
            "sample_ids": list(),
            "sample_names": list(),
            "preds_classif": list(),
            "preds_attr": list()

        }

        for metric_name in self.metric_names:
            if metric_name in self.linked_metrics:
                for linked_metric_name in self.linked_metrics[metric_name]:
                    if linked_metric_name not in self.epoch_metrics.keys():
                        self.epoch_metrics[linked_metric_name] = list()
            else:
                self.epoch_metrics[metric_name] = list()

    def add_batch_values(self, batch_values):
        batch_metrics = self.compute_metrics_from_batch_values(batch_values)
        for key in batch_metrics.keys():
            if key in self.epoch_metrics:
                self.epoch_metrics[key] += batch_metrics[key]

    def compute_metrics_from_batch_values(self, batch_values):
        metrics = dict()
        for v in batch_values.keys():
            metrics[v] = batch_values[v]
        for metric_name in self.metric_names:
            if metric_name == "accuracy":
                metrics[metric_name] = accuracy(metrics["preds_classif"], metrics["gt_classif"])
            elif metric_name == "top5":
                metrics[metric_name] = topk_accuracy(metrics["preds_classif"], metrics["gt_classif"], k=5)
            elif "accuracy_" in metric_name:
                values = metrics[metric_name]
                metrics[metric_name] = accuracy(values, metrics["gt_classif"])
                name = "top5_" + "_".join(metric_name.split("_")[1:])
                metrics[name] = topk_accuracy(values, metrics["gt_classif"], k=5)
            elif metric_name == "attr_mAP":
                metrics = metrics | compute_sample_mAP(metrics["preds_attr"], metrics["gt_attr"], metrics['certainty_masks'])
            elif "attr_mAP_" in metric_name:
                temp = compute_sample_mAP(metrics[metric_name], metrics["gt_attr"], metrics['certainty_masks'])
                metrics["{}_num_attr_by_mask".format(metric_name)] = temp["num_attr_by_mask"]
                metrics["{}_attr_AP_by_mask".format(metric_name)] = temp["attr_AP_by_mask"]
            elif metric_name == "loc_mAP":
                metrics[metric_name] = compute_loc_mAP(metrics["preds_loc"], metrics["gt_loc"])
        return metrics

    def get_display_values(self):
        metric_names = self.metric_names.copy()
        display_values = dict()

        for metric_name in metric_names:
            if metric_name in ["accuracy", "top5"]:
                value = 100 * np.mean(self.epoch_metrics[metric_name])
            elif "accuracy_" in metric_name or 'top5_' in metric_name:
                value = 100 * np.mean(self.epoch_metrics[metric_name])
            elif metric_name in ["attr_mAP", ]:
                value = 100 * compute_mAP_by_class_and_global(self.epoch_metrics["attr_AP_by_mask"], self.epoch_metrics["num_attr_by_mask"], self.epoch_metrics["attr_classes"])
            elif "attr_mAP_" in metric_name:
                value = 100 * compute_mAP_by_class_and_global(self.epoch_metrics["{}_attr_AP_by_mask".format(metric_name)], self.epoch_metrics["{}_num_attr_by_mask".format(metric_name)], self.epoch_metrics["attr_classes"])
            elif metric_name in ["loc_mAP", ]:
                value, _ = compute_loc_mAP_by_attr_and_global(self.epoch_metrics["loc_mAP"])
                value = 100 * value
            elif metric_name in ["loss", "loss_class_image", "loss_attr_image", "loss_loc", "loss_loc_oracle", "loss_class_linear", "loss_attr_linear"]:
                value = None
                if len(self.epoch_metrics[metric_name]) > 0:
                    mask = np.array(self.epoch_metrics[metric_name]) != None
                    weights = np.array(self.epoch_metrics["nb_samples"])[mask]
                    if np.sum(weights) > 0:
                        value = np.average(np.array(self.epoch_metrics[metric_name])[mask], weights=weights)
            else:
                value = self.epoch_metrics[metric_name]

            display_values[metric_name] = round(value, 2) if value is not None else None

        return display_values


def compute_sample_mAP(preds, gt, certainty_masks):
    num_samples = gt.size(0)
    num_masks = certainty_masks.size(0)
    AP = torch.zeros((num_samples, num_masks))
    num_attr = torch.zeros((num_samples, num_masks))
    for i, pred in enumerate(preds):
        # ordering predictions
        pred_attr, order_attr = preds[i].sort(descending=True)
        gt_attr = gt[i, order_attr]
        mask_attr = certainty_masks[:, i, order_attr].bool()
        for m in range(num_masks):
            m_gt = gt_attr[mask_attr[m]]
            num_correct = torch.sum(m_gt)
            if num_correct <= 0:
                continue
            precision = torch.cumsum(m_gt, dim=0) / torch.arange(1, m_gt.size(0)+1)
            recall = torch.cumsum(m_gt, dim=0) / num_correct
            max_precision = torch.cummax(precision.flip(dims=(0, )), dim=0)[0].flip(dims=(0, ))
            shift_recall = torch.clone(recall)
            shift_recall[1:] = shift_recall[:-1].clone()
            shift_recall[0] = 0
            recall_diff = recall - shift_recall
            AP[i, m] = torch.dot(max_precision, recall_diff)
            num_attr[i, m] = m_gt.size(0)
    return {
        "num_attr_by_mask": num_attr.tolist(),
        "attr_AP_by_mask": AP.tolist(),
    }


def compute_mAP_by_class_and_global(AP_by_sample_by_mask, num_attr_by_sample_by_mask, classes):
    num_masks = len(AP_by_sample_by_mask[0])
    classes = torch.tensor(classes)
    AP_by_sample_by_mask = torch.tensor(AP_by_sample_by_mask, dtype=torch.float)
    num_attr_by_sample_by_mask = torch.tensor(num_attr_by_sample_by_mask, dtype=torch.float)
    class_unique = torch.unique(classes)
    AP_by_class_by_mask = dict()
    for class_id in class_unique:
        mask = classes == class_id
        APs = AP_by_sample_by_mask[mask]
        num_attrs = num_attr_by_sample_by_mask[mask]
        total_attrs = torch.sum(num_attrs, dim=0)
        class_AP = torch.bmm(APs.permute(1, 0).unsqueeze(1), num_attrs.permute(1, 0).unsqueeze(2)).view(num_masks) / total_attrs
        class_AP[total_attrs <= 0] = 0
        AP_by_class_by_mask[int(class_id)] = (class_AP, total_attrs)

    APs = torch.stack([AP_by_class_by_mask[k][0] for k in AP_by_class_by_mask.keys()], dim=0)
    num_attrs = torch.stack([AP_by_class_by_mask[k][1] for k in AP_by_class_by_mask.keys()], dim=0)
    total_attrs = torch.sum(num_attrs, dim=0)
    mAP = torch.bmm(APs.permute(1, 0).unsqueeze(1), num_attrs.permute(1, 0).unsqueeze(2)).view(num_masks) / total_attrs

    for k in AP_by_class_by_mask:
        AP_by_class_by_mask[k] = AP_by_class_by_mask[k][0].numpy()

    return mAP[-1].item()


def compute_loc_mAP_by_attr_and_global(list_AP):
    # list_AP (N x A)
    # N: num samples
    # A: nm attributes
    list_AP = np.array(list_AP)
    mask = list_AP == None
    num_AP_by_attr = np.sum(~mask, axis=0)
    list_AP[mask] = 0
    list_AP = list_AP.astype(float)
    mAP_by_attr = np.sum(list_AP, axis=0) / num_AP_by_attr
    mask_nan = np.isnan(mAP_by_attr)
    mAP_by_attr[mask_nan] = 0
    mAP = np.sum(mAP_by_attr) / np.sum(~mask_nan)
    mAP_by_attr = mAP_by_attr.astype(object)
    mAP_by_attr[mask_nan] = None
    return mAP, mAP_by_attr


def compute_loc_mAP(preds, gts):
    preds = preds.permute(0, 2, 1)
    gts = gts.permute(0, 2, 1)
    B, A, L = preds.size()
    AP = compute_batch_mAP(preds.reshape(B * A, L), gts.reshape(B * A, L))
    loc_AP = np.array(AP).reshape((B, A)).tolist()
    return loc_AP


def compute_mAP(scores, gt):
    if torch.sum(gt) == 0:
        return None
    ordered_scores, indices = torch.sort(scores, descending=True)
    ordered_gt = gt[indices]
    num_correct = torch.sum(gt)
    precision = torch.cumsum(ordered_gt, dim=0) / torch.arange(1, ordered_gt.size(0) + 1)
    recall = torch.cumsum(ordered_gt, dim=0) / num_correct
    max_precision = torch.cummax(precision.flip(dims=(0,)), dim=0)[0].flip(dims=(0,))
    shift_recall = torch.clone(recall)
    shift_recall[1:] = shift_recall[:-1].clone()
    shift_recall[0] = 0
    recall_diff = recall - shift_recall
    return float(torch.dot(max_precision, recall_diff))


def compute_batch_mAP(scores, gt):
    ordered_scores, indices = torch.sort(scores, descending=True, dim=1)
    ordered_gt = torch.stack([gt[i][indices[i]] for i in range(gt.size(0))], dim=0)
    num_correct = torch.sum(gt, dim=1)
    precision = torch.cumsum(ordered_gt, dim=1) / torch.arange(1, ordered_gt.size(1) + 1)
    recall = torch.cumsum(ordered_gt, dim=1) / num_correct.unsqueeze(1)
    max_precision = torch.cummax(precision.flip(dims=(1,)), dim=1)[0].flip(dims=(1,))
    shift_recall = torch.clone(recall)
    shift_recall[:, 1:] = shift_recall[:, :-1].clone()
    shift_recall[:, 0] = 0
    recall_diff = recall - shift_recall
    batch_AP = (max_precision.unsqueeze(1) @ recall_diff.unsqueeze(2)).squeeze(2).squeeze(1)
    batch_AP = [float(ap) if c > 0 else None for ap, c in zip(batch_AP, num_correct)]
    return batch_AP


def compute_top1_by_attr_and_global(matching, global_num_attr, k):
    weights_by_attr = np.array([a[k] for a in global_num_attr])
    value_by_attr = np.array([a[k] for a in matching])
    num_attr = weights_by_attr.shape[1]
    avg_by_attr = np.array([np.average([a[i] for a in value_by_attr], weights=[a[i] for a in weights_by_attr]) if np.sum([a[i] for a in weights_by_attr]) > 0 else None for i in range(num_attr)])
    global_weights_by_attr = np.array([np.sum([w[i] for w in weights_by_attr]) for i in range(num_attr)])
    mask = np.array([a != None for a in avg_by_attr])
    global_avg = 100*np.average(avg_by_attr[mask], weights=global_weights_by_attr[mask]) if np.sum(global_weights_by_attr[mask]) > 0 else 0
    avg_by_attr = [100 * a if a is not None else a for a in avg_by_attr]
    avg_weights_by_attr = [np.mean(a[a > 0]) if (a > 0).max() else None for a in weights_by_attr.T]
    return avg_by_attr, avg_weights_by_attr, global_avg


def topk_attr_by_mask(prediction, gt, certainty_masks, k=5):
    batch_size = prediction.size(0)
    num_mask = certainty_masks.size(0)
    num_attr_by_mask = certainty_masks.sum(dim=2)
    masked_similarity = certainty_masks * prediction
    topk = masked_similarity.topk(k, dim=2)[1]
    masked_matching = torch.zeros(topk.size())
    metrics = dict()
    for k in range(num_mask):
        for i in range(batch_size):
            masked_matching[k][i] = gt[i].index_select(0, topk[k, i])
    metrics["attr_top1_by_mask"] = masked_matching[:, :, 0].permute(1, 0).tolist()
    metrics["attr_top5_by_mask"] = torch.mean(masked_matching, dim=2).permute(1, 0).tolist()
    metrics["num_attr_by_mask"] = num_attr_by_mask.permute(1, 0).tolist()
    return metrics


def topk_attr_mul_by_mask(attr_bin_mul_mapping, prediction, gt, certainty_masks):
    batch_size = prediction.size(0)
    num_mask = certainty_masks.size(0)
    masked_similarity = certainty_masks * prediction
    gt_by_mul_attr = [gt.index_select(dim=1, index=torch.tensor(attr_bin_mul_mapping[mul_id])) for mul_id in sorted(attr_bin_mul_mapping.keys())]
    gt_num_correct_by_mask = torch.stack([torch.sum(gt_mul, dim=1)for gt_mul in gt_by_mul_attr], dim=0).permute(1, 0)
    similarity_by_mul_attr_by_mask = [masked_similarity.index_select(dim=2, index=torch.tensor(attr_bin_mul_mapping[mul_id])) for mul_id in sorted(attr_bin_mul_mapping.keys())]
    num_attr_by_mask = torch.stack([(sim != 0).sum(dim=2) for sim in similarity_by_mul_attr_by_mask], dim=2)
    num_attr = len(gt_by_mul_attr)
    matching = torch.zeros((num_mask, batch_size, num_attr))

    for m in range(num_mask):
        for b in range(batch_size):
            for a in range(num_attr):
                matching[m, b, a] = gt_by_mul_attr[a][b, similarity_by_mul_attr_by_mask[a][m, b, :].argmax()]
    metrics = dict()
    metrics["attr_mul_top1_by_mask"] = [matching[:, b, :].tolist() for b in range(batch_size)]
    metrics["num_attr_mul_by_attr_by_mask"] = [(torch.stack([torch.logical_and(gt_num_correct_by_mask[b] >= 1, num_attr_by_mask[m, b] >= 1) for m in range(num_mask)] , dim=0) * num_attr_by_mask[:, b, :]).tolist() for b in range(batch_size)]
    return metrics


def accuracy(prediction, gt):
    return (torch.argmax(prediction, dim=1) == gt).int().numpy().tolist()


def topk_accuracy(prediction, gt, k=5):
    return torch.sum(prediction.topk(k, dim=1)[1] == gt.unsqueeze(1), dim=1).numpy().tolist()

