import torch
import numpy as np
import lib.utils.tensor_utils as tensor_utils
import lib.embedding.lda as lda
import lib.evaluation.knn_eval as knn_eval


def lda_eval(x_data, y_data, x_test, y_test, args, logger, has_same=False):
    logger.info(f"start knn(k={args.TEST.neighbor_k}) ... ")

    x_data = tensor_utils.numpy_to_tensor(x_data)
    x_test = tensor_utils.numpy_to_tensor(x_test)

    y_data = tensor_utils.numpy_to_tensor(y_data)
    y_test = tensor_utils.numpy_to_tensor(y_test)

    with torch.no_grad():
        logits, labels, v = lda.lda_prediction_main(
            train_feats=x_data,
            train_labels=y_data,
            test_feats=x_test,
            test_labels=y_test,
            need_v=True
        )

        transformed_train_feats = torch.mm(x_data, v)
        transformed_test_feats = torch.mm(x_test, v)

        # print(transformed_train_feats.shape)
        # print(transformed_test_feats.shape)
        # sfa

        knn_scores = knn_eval.knn_eval(
            x_data=transformed_train_feats,
            y_data=y_data,
            x_test=transformed_test_feats,
            y_test=y_test,
            args=args,
            logger=logger
        )

        _, pred = torch.max(logits, dim=1)
        acc = np.mean(pred.numpy() == labels.numpy())

    scores = {
        "top1_acc": acc
    }

    for key, score in knn_scores.items():
        scores["{}_projected".format(key)] = score

    return scores
