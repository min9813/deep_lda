import torch
import torch.nn as nn
import torch.nn.functional as F


class LossWrap(torch.nn.Module):
    def __init__(self, args, model, criterion, head, embedder=None, embedder_test=None):
        self.args = args
        super(LossWrap, self).__init__()
        self.model = model
        self.criterion = criterion
        self.head = head

        self.embedder = embedder
        self.embedder_test = embedder_test

    def forward(self, input, label):
        if self.args.multi_gpus:
            input, label = input.cuda(), label.cuda()
        else:
            input, label = input.to(
                self.args.device), label.to(self.args.device)

        features = self.model(input)
        logits = self.head(features)

        if self.args.TRAIN.need_cls_loss:
            cls_loss = self.criterion(logits, label)
        else:
            cls_loss = torch.tensor(0., device=logits.device, dtype=logits.dtype)
            
        output = {
            "logit": logits,
            "loss_cls": cls_loss
        }

        if self.embedder is not None:
            try:
                embed_loss, w, v = self.embedder(features, label, lamb=0.001)
                ok = 1

            except RuntimeError:
                if torch.any(torch.isnan(logits)):
                    print("has nan in logits")
                if torch.any(torch.isnan(features)):
                    print("has nan in features")
                embed_loss = torch.tensor(0, device=features.device, dtype=features.dtype)
                lda_cls_loss = torch.tensor(0, device=features.device, dtype=features.dtype)
                lda_acc = torch.tensor(0, device=features.device, dtype=features.dtype)
                ok = 0
                sfdalamb

            if ok > 0.5:
                if self.embedder_test is not None:
                    logit, this_labels = self.embedder_test(
                        train_feats=features,
                        train_labels=label,
                        # test_h=features,
                        # test_l=label,
                        # v=v
                    )
                    # logit = logit.clamp(min=5e-4, max=1-5e-4).log()
                    # print(logit.shape)
                    # print(this_labels)
                    # sfa
                    lda_cls_loss = self.criterion(logit, this_labels)

                    with torch.no_grad():
                        _, pred = torch.max(logit, dim=1)
                        lda_acc = torch.sum(pred==this_labels) / pred.shape[0]
                    # lda_cls_loss = F.nll_loss(logit, label)
                    
                    # print(lda_cls_loss)
                    # print(logit[:10])
                    # print(label[:10])
                    # sfdalamb
                else:
                    lda_cls_loss = torch.tensor(0, device=features.device, dtype=features.dtype)
                    lda_acc = torch.tensor(0, device=features.device, dtype=features.dtype)

            output["loss_embed"] = embed_loss
            output["lda_acc"] = lda_acc
            output["loss_lda_cls_loss"] = lda_cls_loss
            output["embed_ok"] = ok

            output["loss_total"] = embed_loss + cls_loss + lda_cls_loss
            # output["loss_total"] = embed_loss

        else:
            output["loss_total"] = cls_loss

        if self.args.TEST.mode == "knn":
            output["features"] = features

        return output
