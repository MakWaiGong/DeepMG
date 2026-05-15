import json
import torch
import torch.profiler
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
from models import DeepMG
import numpy as np
import random
import time
import socket
from torchmetrics import AUROC,Accuracy,F1Score,ConfusionMatrix,MatthewsCorrCoef,Precision,Recall,AveragePrecision
import copy
from torch.multiprocessing import spawn
import json
import os
from torch.utils.tensorboard import SummaryWriter
from preprocess import load_data
import warnings
from sklearn.metrics import roc_auc_score,confusion_matrix,precision_recall_curve,auc
import math
import gc
import psutil


def clear_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()

if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("none")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def score1(y_test, y_pred):
    auc_roc_score = roc_auc_score(y_test, y_pred)
    prec, recall, _ = precision_recall_curve(y_test, y_pred)
    prauc = auc(recall, prec)
    y_pred_print = [round(y, 0) for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_print).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
    P = tp / (tp + fp)
    F1 = (P * se * 2) / (P + se)
    BA = (se + sp) / 2
    PPV = tp / (tp + fp)
    NPV = tn / (fn + tn)
    metrics = {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Sensitivity': se,
        'Specificity': sp,
        'MCC': mcc,
        'ACC': acc,
        'AUROC': auc_roc_score,
        'F1': F1,
        'BA': BA,
        'AUPRC': prauc,
        'PPV': PPV,
        'NPV': NPV
    }
    return metrics

def seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)


def main(rank, world_size, start_time, master_port, params):
    seed(params["seed"])
    if rank <= 0:
        if os.access("history.json", os.R_OK):
            with open("history.json", "r", encoding="utf-8") as f:
                j = json.load(f)
        else:
            j = {}
        meta = {"Hostname": socket.gethostname(), "Worldsize": world_size}
        j[start_time] = {**meta, **params}
        with open("history.json", "w", encoding="utf-8") as f:
            json.dump(j, f, indent=4, ensure_ascii=False)

        writer = SummaryWriter("./logs/" + start_time)
    else:
        writer = None

    if rank == -1:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = master_port
        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        device = torch.device("cuda", rank)
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    test_metrics,score = train(rank, device, writer, params, start_time)
    if rank <= 0:
        writer.close()
        with open("history.json", "r", encoding="utf-8") as f:
            j = json.load(f)
        for k in test_metrics.keys():
            j[start_time].update({f"Test{k}": test_metrics[k]})

        for k in score.keys():
            j[start_time].update({f"Score{k}": float(score[k]) if isinstance(score[k], np.int64) else score[k]})

        with open("history.json", "w", encoding="utf-8") as f:
            json.dump(j, f, indent=4, ensure_ascii=False)



def sanity_check_graph(data, name="graph"):
    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    N = x.size(0)
    E = edge_index.size(1)
    print(f"[{name}] nodes={N}, feat_dim={x.size(1)}, edges={E}, edge_attr_shape={None if edge_attr is None else tuple(edge_attr.shape)}")

    # 1) index range
    if edge_index.numel() == 0:
        print(f"[{name}] WARNING: empty edge_index")
    else:
        imin = int(edge_index.min().item())
        imax = int(edge_index.max().item())
        assert imin >= 0, f"[{name}] edge_index has negative indices: min={imin}"
        assert imax < N, f"[{name}] edge_index max index {imax} >= num_nodes {N} !!"

    # 2) edge_attr length matches E
    if edge_attr is not None:
        assert edge_attr.size(0) == E, f"[{name}] edge_attr first dim {edge_attr.size(0)} != num_edges {E}"

    # 3) device & dtype checks
    dev = x.device
    if edge_index.device != dev:
        raise AssertionError(f"[{name}] edge_index device {edge_index.device} != x device {dev}")
    if edge_attr is not None and edge_attr.device != dev:
        raise AssertionError(f"[{name}] edge_attr device {edge_attr.device} != x device {dev}")

    # 4) dtypes
    assert edge_index.dtype == torch.long, f"[{name}] edge_index dtype should be torch.long but is {edge_index.dtype}"
    if edge_attr is not None and not torch.is_floating_point(edge_attr):
        print(f"[{name}] WARNING: edge_attr is not float (dtype={edge_attr.dtype}); converting to float may be needed")


def train(local_rank, device, writer, params, start_time):
    with_valid = params["valid"]
    if with_valid:
        train_data, valid_data, test_data = load_data(params)
    else:
        train_data, test_data = load_data(params)

    model = DeepMG()
    model.to(device)
    if local_rank != -1:
        model = DDP(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=params["learning_rate"])

    if local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data, shuffle=True
        )
        train_loader = DataLoader(
            train_data,
            batch_size=params["batch_size"],
            sampler=train_sampler,
            pin_memory=True,
            num_workers=1,
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
        test_loader = DataLoader(
            test_data,
            batch_size=params["batch_size"],
            sampler=test_sampler,
            pin_memory=True,
            num_workers=1,
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=params["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        )
        test_loader = DataLoader(
            test_data, batch_size=params["batch_size"], pin_memory=True, num_workers=1
        )

    if with_valid:
        valid_bce_obj = nn.BCELoss().to(device)
        if local_rank != -1:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)
            valid_loader = DataLoader(
                valid_data,
                batch_size=params["batch_size"],
                sampler=valid_sampler,
                pin_memory=True,
                num_workers=1,
            )
        else:
            valid_loader = DataLoader(
                valid_data,
                batch_size=params["batch_size"],
                pin_memory=True,
                num_workers=1,
            )

    best_mse = 1000
    best_epoch = -1
    acc_s = []
    loss_s = []
    for epoch in range(1, params["max_epoch"] + 1):

        if local_rank != -1:
            train_loader.sampler.set_epoch(epoch)
            if with_valid:
                valid_loader.sampler.set_epoch(epoch)

        model.train()
        loss_list = []
        for _, data in enumerate(train_loader):
            drug_graph_e3 = data[0].to(device)
            target_graph_e3 = data[1].to(device)
            target_graph_poi = data[2].to(device)

            output = model(drug_graph_e3, target_graph_e3, target_graph_poi).view(-1)

            Y = data[3].to(device).view(-1)

            loss = loss_fn(output, Y)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if writer:
            writer.add_scalar("Train Loss", np.mean(loss_list), epoch)

        if with_valid:
            if epoch % 1 == 0 :
                if local_rank != -1:
                    dist.barrier()
                model.eval()
                with torch.no_grad():
                    valid_loss = 0.0
                    total_samples = 0
                    for data in valid_loader:
                        drug_graph_e3 = data[0].to(device)
                        target_graph_e3 = data[1].to(device)
                        target_graph_poi = data[2].to(device)

                        output = model(drug_graph_e3, target_graph_e3, target_graph_poi).view(-1)

                        Y = data[3].to(device).view(-1)

                        valid_bce=valid_bce_obj(output, Y)
                        valid_loss = valid_loss + valid_bce.item()
                        total_samples += Y.size(0)
                    average_valid_loss = valid_loss / total_samples

                if local_rank != -1:
                    dist.barrier()

                if writer:
                    writer.add_scalar("Valid Loss", valid_mse, epoch)
                    writer.flush()
                if average_valid_loss < best_mse:
                    best_mse = average_valid_loss
                    best_epoch = epoch
                    if local_rank == 0:
                        torch.save(model.module.state_dict(), f'./pts/{params["dataset"]}_seed_{params["seed"]}_fold_{params["fold"]}.pt')
                    elif local_rank == -1:
                        torch.save(model.state_dict(), f'./pts/{params["dataset"]}_seed_{params["seed"]}_fold_{params["fold"]}.pt')
                    state_dict_for_test = copy.deepcopy(model.state_dict())
                if local_rank != -1:
                    dist.barrier()
                if epoch - best_epoch >= 30:
                    print(f"Early stopping at epoch {epoch}, no improvement for 30 epochs.")
                    break

        print(f'fold:{params["fold"]},Epoch: {epoch}/1000, Train Loss :{np.mean(loss_list):.4f},"Valid Loss", {average_valid_loss:.4f},best_epoch:{best_epoch}.')

    if local_rank != -1:
        dist.barrier()
    warnings.filterwarnings("ignore", category=UserWarning)
    metrics = {
        "AUROC": AUROC(task='binary').to(device),
        "ACC": Accuracy(task='binary').to(device),
        "F1": F1Score(task='binary').to(device),
        "AUPRC": AveragePrecision(task='binary').to(device),
        "Precision": Precision(task='binary').to(device),
        "Recall": Recall(task='binary').to(device),
        "ConfusionMatrix":ConfusionMatrix(num_classes=2,task='binary').to(device),
        "MCC": MatthewsCorrCoef(task="binary").to(device)
    }
    warnings.resetwarnings()
    if "state_dict_for_test" in locals():
        model.load_state_dict(state_dict_for_test)
    model.eval()
    preds=[]
    target=[]
    with torch.no_grad():
        for data in test_loader:
            drug_graph_e3 = data[0].to(device)
            target_graph_e3 = data[1].to(device)
            target_graph_poi = data[2].to(device)
            output = model(drug_graph_e3, target_graph_e3, target_graph_poi).view(-1)

            preds = preds+output.tolist()
            Y = data[3].to(device).view(-1).to(torch.int64)
            target = target + Y.tolist()

            for m in metrics.values():
                m.update(output, Y)

        SCORE = score1(target, preds)
        print(f"SCORE:{SCORE}")



    if local_rank != -1:
        dist.barrier()
    vals = {}
    for k, v in metrics.items():
        if k == "ConfusionMatrix":
            cm = v.compute()
            print(cm)
        else:
            vals[k] = float(v.compute())
            v.reset()

    vals['TN'] = len([i for i in range(len(preds)) if preds[i] < 0.5 and target[i] == 0])
    vals['TP'] = len([i for i in range(len(preds)) if preds[i] >= 0.5 and target[i] == 1])
    vals['FN'] = len([i for i in range(len(preds)) if preds[i] < 0.5 and target[i] == 1])
    vals['FP'] = len([i for i in range(len(preds)) if preds[i] >= 0.5 and target[i] == 0])

    print(f"vals:{vals}")

    if local_rank != -1:
        dist.barrier()
        dist.destroy_process_group()
    return vals,SCORE


def run(params):
    world_size = torch.cuda.device_count()
    start_time = time.strftime("%m%d-%H%M%S", time.localtime())
    master_port = str(random.randint(30000, 39999))
    if world_size > 1:
        spawn(
            main,
            args=(world_size, start_time, master_port, params),
            nprocs=world_size,
            join=True,
        )
    else:
        main(-1, world_size, start_time, None, params)

    with open("history.json", "r", encoding="utf-8") as f:
            j = json.load(f)

    return (j[start_time]["TestAUROC"], j[start_time]["TestACC"], j[start_time]["TestF1"],j[start_time]["TestAUPRC"],
            j[start_time]["TestPrecision"],j[start_time]["TestRecall"],j[start_time]["TestMCC"],
            j[start_time]["TestTN"],j[start_time]["TestTP"],j[start_time]["TestFN"],j[start_time]["TestFP"],
            j[start_time]["ScoreTN"], j[start_time]["ScoreTP"], j[start_time]["ScoreFN"], j[start_time]["ScoreFP"],
            j[start_time]["ScoreSensitivity"], j[start_time]["ScoreSpecificity"], j[start_time]["ScoreMCC"],
            j[start_time]["ScoreACC"], j[start_time]["ScoreAUROC"], j[start_time]["ScoreF1"], j[start_time]["ScoreBA"],
            j[start_time]["ScoreAUPRC"], j[start_time]["ScorePPV"], j[start_time]["ScoreNPV"])



if __name__ == "__main__":
    params = {
        "dataset": "MG_Nodes",
        "valid": True,
        "seed": 40,
        "max_epoch": 1000,
        "batch_size": 16,
        "learning_rate": 0.001,
    }
    l_auroc=[]
    l_acc=[]
    l_f1=[]
    l_auprc = []
    l_pression = []
    l_recall = []
    l_mcc = []
    l_tn=[]
    l_tp=[]
    l_fn=[]
    l_fp=[]


    l1_TN = []
    l1_TP = []
    l1_FN = []
    l1_FP = []
    l1_Se = []
    l1_Sp = []
    l1_F1 = []
    l1_MCC = []
    l1_ACC = []
    l1_AUROC = []
    l1_BA = []
    l1_AUPRC = []
    l1_PPV = []
    l1_NPV = []
    for idx in range(0, 5):    # 5-fold cross-validation
        params["fold"] = idx
        score={}
        print("Memory usage (MB)_fold:", psutil.Process().memory_info().rss / 1024 / 1024)
        clear_cuda_memory()
        print("Memory usage (MB)_fold1:", psutil.Process().memory_info().rss / 1024 / 1024)
        auroc, acc, f1, auprc, pression, recall, mcc,tn, tp, fn, fp,Tn, Tp, Fn, Fp, SES, SPS, Mcc, Acc, AUroc, F11, Ba, AUprc, Ppv, Npv= run(params)

        l_auroc.append(auroc)
        l_acc.append(acc)
        l_f1.append(f1)
        l_auprc.append(auprc)
        l_pression.append(pression)
        l_mcc.append(mcc)
        l_tn.append(tn)
        l_tp.append(tp)
        l_fn.append(fn)
        l_fp.append(fp)

        l1_TN.append(Tn)
        l1_TP.append(Tp)
        l1_FN.append(Fn)
        l1_FP.append(Fp)
        l1_Se.append(SES)
        l1_Sp.append(SPS)
        l1_F1.append(F11)
        l1_MCC.append(Mcc)
        l1_ACC.append(Acc)
        l1_AUROC.append(AUroc)
        l1_BA.append(Ba)
        l1_AUPRC.append(AUprc)
        l1_PPV.append(Ppv)
        l1_NPV.append(Npv)


    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(f"AUROC: {np.mean(l_auroc):.4f} ({np.std(l_auroc):.4f})")
    print(f"ACC: {np.mean(l_acc):.4f} ({np.std(l_acc):.4f})")
    print(f"F1: {np.mean(l_f1):.4f} ({np.std(l_f1):.4f})")
    print(f"AUPRC: {np.mean(l_auprc):.4f} ({np.std(l_auprc):.4f})")
    print(f"Pression: {np.mean(l_pression):.4f} ({np.std(l_pression):.4f})")
    print(f"Recall: {np.mean(l_recall):.4f} ({np.std(l_recall):.4f})")
    print(f"MCC: {np.mean(l_mcc):.4f} ({np.std(l_mcc):.4f})")
    print(f"TN: {np.mean(l_tn):.4f} ({np.std(l_tn):.4f})")
    print(f"TP: {np.mean(l_tp):.4f} ({np.std(l_tp):.4f})")
    print(f"FN: {np.mean(l_fn):.4f} ({np.std(l_fn):.4f})")
    print(f"FP: {np.mean(l_fp):.4f} ({np.std(l_fp):.4f})")

    print("SCORE:")
    print(f"TN_1: {np.mean(l1_TN):.4f} ({np.std(l1_TN):.4f})")
    print(f"TP_1: {np.mean(l1_TP):.4f} ({np.std(l1_TP):.4f})")
    print(f"FN_1: {np.mean(l1_FN):.4f} ({np.std(l1_FN):.4f})")
    print(f"FP_1: {np.mean(l1_FP):.4f} ({np.std(l1_FP):.4f})")
    print(f"Se: {np.mean(l1_Se):.4f} ({np.std(l1_Se):.4f})")
    print(f"Sp: {np.mean(l1_Sp):.4f} ({np.std(l1_Sp):.4f})")
    print(f"MCC: {np.mean(l1_MCC):.4f} ({np.std(l1_MCC):.4f})")
    print(f"ACC: {np.mean(l1_ACC):.4f} ({np.std(l1_ACC):.4f})")
    print(f"AUROC: {np.mean(l1_AUROC):.4f} ({np.std(l1_AUROC):.4f})")
    print(f"F1: {np.mean(l1_F1):.4f} ({np.std(l1_F1):.4f})")
    print(f"BA: {np.mean(l1_BA):.4f} ({np.std(l1_BA):.4f})")
    print(f"AUPRC: {np.mean(l1_AUPRC):.4f} ({np.std(l1_AUPRC):.4f})")
    print(f"PPV: {np.mean(l1_PPV):.4f} ({np.std(l1_PPV):.4f})")
    print(f"NPV: {np.mean(l1_NPV):.4f} ({np.std(l1_NPV):.4f})")
