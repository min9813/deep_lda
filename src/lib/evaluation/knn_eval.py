import torch
import lib.utils.tensor_utils as tensor_utils
import lib.evaluation.metric as metric


def knn_eval(x_data, y_data, x_test, y_test, args, logger, has_same=False):
    logger.info(f"start knn(k={args.TEST.neighbor_k}) ... ")

    x_data = tensor_utils.numpy_to_tensor(x_data)
    x_test = tensor_utils.numpy_to_tensor(x_test)

    y_data = tensor_utils.numpy_to_tensor(y_data)
    y_test = tensor_utils.numpy_to_tensor(y_test)

    top1_acc = 0
    top5_acc = 0
    num_classes = y_data.max() + 1
    # x_data = x_data / \
    #     torch.sqrt(torch.sum(x_data * x_data, axis=-1, keepdim=True))
    # # x_test = x_data
    # x_test = x_test / \
    #     torch.sqrt(torch.sum(x_test * x_test, axis=-1, keepdim=True))
    BT = x_test.size(0)
    batch_num = (BT+(args.TEST.batch_size-1)) // args.TEST.batch_size
    num = 0
    for batch_idx in range(batch_num):
        batch_test = x_test[batch_idx *
                            args.TEST.batch_size: (batch_idx+1)*args.TEST.batch_size]
        batch_test_l = y_test[batch_idx *
                              args.TEST.batch_size: (batch_idx+1)*args.TEST.batch_size]
        bsize = batch_test.size(0)
        num += bsize
        # sim_matrix = torch.mm(batch_test, x_data.T)
        sim_matrix = metric.calc_l2_dist_torch(
            feature1=batch_test,
            feature2=x_data,
            is_sqrt=False,
            is_neg=True,
            dim=1
        )
        args.TEST.test_logit_scale = 1
        sim_weight, sim_indices = sim_matrix.topk(
            k=args.TEST.neighbor_k+args.TEST.has_same, dim=-1)
        # print("before:", args.TEST.neighbor_k, sim_indices.size(), args.has_same)
        # print(sim_weight[:30, :2], sim_indices[:30, :2])
        sim_weight = sim_weight[:, int(args.TEST.has_same):]
        sim_indices = sim_indices[:, int(args.TEST.has_same):]
        # print("after", args.TEST.neighbor_k, sim_indices.size(), args.has_same)
        # print(sim_weight[:30, :2], sim_indices[:30, :2])
        # knjdfklsgjkl
        # [B, K]
        sim_labels = torch.gather(y_data.expand(
            bsize, -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight * args.TEST.test_logit_scale).exp()
        
        # print("sim label:", sim_labels.size())

        # counts for each class
        one_hot_label = torch.zeros(
            bsize * args.TEST.neighbor_k, num_classes, device=sim_labels.device)
        # print("one hot label:", one_hot_label.size())
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(
            dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(
            bsize, -1, num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        top1_acc += torch.sum((pred_labels[:, :1] == batch_test_l.unsqueeze(
            dim=-1)).any(dim=-1).float()).item()
        top5_acc += torch.sum((pred_labels[:, :5] == batch_test_l.unsqueeze(
            dim=-1)).any(dim=-1).float()).item()
    # print(num, BT, batch_num, BT+(args.TEST.batch_size-1), args.TEST.batch_size)
    top1_acc /= BT
    top5_acc /= BT
    result = {
        "top1": top1_acc,
        "top5": top5_acc
    }
    # torch.gather(x_data, dim=1, index=indice)

    return result
