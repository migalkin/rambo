import wandb
from tqdm import tqdm
from typing import Callable

from mytorch import dataiters
from mytorch.utils.goodies import *

from corruption import Corruption


# Make a loop fn
def training_loop(epochs: int,
                  data: dict,
                  opt: torch.optim,
                  train_fn: Callable,
                  neg_generator: Corruption,
                  device: torch.device = torch.device('cpu'),
                  data_fn: Callable = dataiters.SimplestSampler,
                  eval_fn_trn: Callable = default_eval,
                  val_testbench: Callable = default_eval,
                  trn_testbench: Callable = default_eval,
                  eval_every: int = 1,
                  log_wandb: bool = True,
                  run_trn_testbench: bool = True) -> (list, list, list):
    """
        A fn which can be used to train a language model.

        The model doesn't need to be an nn.Module,
            but have an eval (optional), a train and a predict function.

        Data should be a dict like so:
            {"train":{"x":np.arr, "y":np.arr}, "val":{"x":np.arr, "y":np.arr} }

        Train_fn must return both loss and y_pred

        :param epochs: integer number of epochs
        :param data: a dictionary which looks like {'train': train data}
        :param opt: torch optimizer
        :param train_fn: a fn which is/can call forward of a nn module
        :param neg_generator: A corruption instance which can be used to corrupt one batch of pos data
        :param device: torch.device for making tensors
        :param data_fn: Something that can make iterators out of training data (think mytorch samplers)
        :param eval_fn_trn: Function which can take a bunch of pos, neg scores and give out some metrics
        :param val_testbench: Function call to see generate all negs for all pos and get metrics in valid set
        :param trn_testbench:Function call to see generate all negs for all pos and get metrics in train set
        :param eval_every: int which dictates after how many epochs should run testbenches
        :param log_wandb: bool which dictates whether we log things with wandb
        :param run_trn_testbench: bool which dictates whether we run testbench on train set
    """

    train_loss = []
    train_acc = []
    valid_acc = []
    valid_mrr = []
    valid_mr = []
    valid_hits_3, valid_hits_5, valid_hits_10 = [], [], []
    train_acc_bnchmk = []
    train_mrr_bnchmk = []
    train_mr_bnchmk = []
    train_hits_3_bnchmk, train_hits_5_bnchmk, train_hits_10_bnchmk = [], [], []
    lrs = []

    # Epoch level
    for e in range(epochs):

        per_epoch_loss = []
        per_epoch_tr_acc = []

        # Train
        with Timer() as timer:

            # Make data
            trn_dl = data_fn(data['train'])

            for pos in tqdm(trn_dl):
                neg = neg_generator.corrupt_batch(pos)
                opt.zero_grad()

                _pos = torch.tensor(pos, dtype=torch.long, device=device)
                _neg = torch.tensor(neg, dtype=torch.long, device=device)

                (pos_scores, neg_scores), loss = train_fn(_pos, _neg)

                per_epoch_tr_acc.append(eval_fn_trn(pos_scores=pos_scores, neg_scores=neg_scores))
                per_epoch_loss.append(loss.item())

                loss.backward()
                opt.step()

        # Log this stuff
        train_acc.append(np.mean(per_epoch_tr_acc))
        train_loss.append(np.mean(per_epoch_loss))

        """
            # Val
            If eval_every enables us to,
                -> we call the partial fn which can gen all negs for each pos and get the metrics on val set
                -> same for the train set
            
        """
        if e % eval_every == 0 and e >= 1:
            with torch.no_grad():
                summary_val = val_testbench()
                per_epoch_vl_acc = summary_val['metrics']['acc']
                per_epoch_vl_mrr = summary_val['metrics']['mrr']
                per_epoch_vl_mr = summary_val['metrics']['mr']
                per_epoch_vl_hits_3 = summary_val['metrics']['hits_at 3']
                per_epoch_vl_hits_5 = summary_val['metrics']['hits_at 5']
                per_epoch_vl_hits_10 = summary_val['metrics']['hits_at 10']

                valid_acc.append(per_epoch_vl_acc)
                valid_mrr.append(per_epoch_vl_mrr)
                valid_mr.append(per_epoch_vl_mr)
                valid_hits_3.append(per_epoch_vl_hits_3)
                valid_hits_5.append(per_epoch_vl_hits_5)
                valid_hits_10.append(per_epoch_vl_hits_10)

                if run_trn_testbench:
                    # Also run train testbench
                    summary_trn = trn_testbench()
                    per_epoch_tr_acc_bnchmk = summary_trn['metrics']['acc']
                    per_epoch_tr_mrr_bnchmk = summary_trn['metrics']['mrr']
                    per_epoch_tr_mr_bnchmk = summary_trn['metrics']['mr']
                    per_epoch_tr_hits_3_bnchmk = summary_trn['metrics']['hits_at 3']
                    per_epoch_tr_hits_5_bnchmk = summary_trn['metrics']['hits_at 5']
                    per_epoch_tr_hits_10_bnchmk = summary_trn['metrics']['hits_at 10']

                    train_acc_bnchmk.append(per_epoch_tr_acc_bnchmk)
                    train_mrr_bnchmk.append(per_epoch_tr_mrr_bnchmk)
                    train_mr_bnchmk.append(per_epoch_tr_mr_bnchmk)
                    train_hits_3_bnchmk.append(per_epoch_tr_hits_3_bnchmk)
                    train_hits_5_bnchmk.append(per_epoch_tr_hits_5_bnchmk)
                    train_hits_10_bnchmk.append(per_epoch_tr_hits_10_bnchmk)

                    # Print statement here
                    print("Epoch: %(epo)03d | Loss: %(loss).5f | Tr_c: %(tracc)0.5f | "
                          "Vl_c: %(vlacc)0.5f | Vl_mrr: %(vlmrr)0.5f | Vl_mr: %(vlmr)0.5f | "
                          "Vl_h3: %(vlh3)0.5f | Vl_h5: %(vlh5)0.5f | Vl_h10: %(vlh10)0.5f | "
                          "Tr_c_b: %(tracc_b)0.5f | Tr_mrr_b: %(trmrr_b)0.5f | Tr_mr_b: %(trmr_b)0.5f | "
                          "Tr_h3_b: %(trh3_b)0.5f | Tr_h5_b: %(trh5_b)0.5f | Tr_h10_b: %(trh10_b)0.5f | "
                          "Time_trn: %(time).3f min"
                          % {'epo': e,
                             'loss': float(np.mean(per_epoch_loss)),
                             'tracc': float(np.mean(per_epoch_tr_acc)),
                             'vlacc': float(per_epoch_vl_acc),
                             'vlmrr': float(per_epoch_vl_mrr),
                             'vlmr': float(per_epoch_vl_mr),
                             'vlh3': float(per_epoch_vl_hits_3),
                             'vlh5': float(per_epoch_vl_hits_5),
                             'vlh10': float(per_epoch_vl_hits_10),
                             'tracc_b': float(per_epoch_tr_acc_bnchmk),
                             'trmrr_b': float(per_epoch_tr_mrr_bnchmk),
                             'trmr_b': float(per_epoch_tr_mr_bnchmk),
                             'trh3_b': float(per_epoch_tr_hits_3_bnchmk),
                             'trh5_b': float(per_epoch_tr_hits_5_bnchmk),
                             'trh10_b': float(per_epoch_tr_hits_10_bnchmk),
                             'time': timer.interval / 60.0})

                    if log_wandb:
                        # Wandb stuff
                        wandb.log({
                            'epoch': e,
                            'loss': float(np.mean(per_epoch_loss)),
                            'trn_acc': float(np.mean(per_epoch_tr_acc)),
                            'val_acc': float(per_epoch_vl_acc),
                            'val_mrr': float(per_epoch_vl_mrr),
                            'val_mr': float(per_epoch_vl_mr),
                            'val_hits@3': float(per_epoch_vl_hits_3),
                            'val_hits@5': float(per_epoch_vl_hits_5),
                            'val_hits@10': float(per_epoch_vl_hits_10),
                            'trn_acc_b': float(per_epoch_tr_acc_bnchmk),
                            'trn_mrr_b': float(per_epoch_tr_mrr_bnchmk),
                            'trn_mr_b': float(per_epoch_tr_mr_bnchmk),
                            'trn_hits@3_b': float(per_epoch_tr_hits_3_bnchmk),
                            'trn_hits@5_b': float(per_epoch_tr_hits_5_bnchmk),
                            'trn_hits@10_b': float(per_epoch_tr_hits_10_bnchmk),
                        })

                else:
                    # Don't benchmark over train

                    # Print Statement here
                    print("Epoch: %(epo)03d | Loss: %(loss).5f | Tr_c: %(tracc)0.5f | "
                          "Vl_c: %(vlacc)0.5f | Vl_mrr: %(vlmrr)0.5f | Vl_mr: %(vlmr)0.5f | "
                          "Vl_h3: %(vlh3)0.5f | Vl_h5: %(vlh5)0.5f | Vl_h10: %(vlh10)0.5f | "
                          "time_trn: %(time).3f min"
                          % {'epo': e,
                             'loss': float(np.mean(per_epoch_loss)),
                             'tracc': float(np.mean(per_epoch_tr_acc)),
                             'vlacc': float(per_epoch_vl_acc),
                             'vlmrr': float(per_epoch_vl_mrr),
                             'vlmr': float(per_epoch_vl_mr),
                             'vlh3': float(per_epoch_vl_hits_3),
                             'vlh5': float(per_epoch_vl_hits_5),
                             'vlh10': float(per_epoch_vl_hits_10),
                             'time': timer.interval / 60.0})

                    if log_wandb:
                        # Wandb stuff
                        wandb.log({
                            'epoch': e,
                            'loss': float(np.mean(per_epoch_loss)),
                            'trn_acc': float(np.mean(per_epoch_tr_acc)),
                            'val_acc_b': float(per_epoch_vl_acc),
                            'val_mrr_b': float(per_epoch_vl_mrr),
                            'val_mr_b': float(per_epoch_vl_mr),
                            'val_hits@3_b': float(per_epoch_vl_hits_3),
                            'val_hits@5_b': float(per_epoch_vl_hits_5),
                            'val_hits@10_b': float(per_epoch_vl_hits_10),
                        })

        else:
            # No test benches this time around
            print("Epoch: %(epo)03d | Loss: %(loss).5f | Tr_c: %(tracc)0.5f | "
                  "Time_Train: %(time).3f min"
                  % {'epo': e,
                     'loss': float(np.mean(per_epoch_loss)),
                     'tracc': float(np.mean(per_epoch_tr_acc)),
                     'time': timer.interval / 60.0})

            if log_wandb:
                # Wandb stuff
                wandb.log({
                    'epoch': e,
                    'loss': float(np.mean(per_epoch_loss)),
                    'trn_acc': float(np.mean(per_epoch_tr_acc))
                })

    return train_acc, train_loss, \
           train_acc_bnchmk, train_mrr_bnchmk, \
           train_hits_3_bnchmk, train_hits_5_bnchmk, train_hits_10_bnchmk, \
           valid_acc, valid_mrr, \
           valid_hits_3, valid_hits_5, valid_hits_10
