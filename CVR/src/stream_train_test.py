from models import get_model
from loss import get_loss_fn
from utils import get_optimizer, ScalarMovingAverage
from metrics import cal_llloss_with_logits, cal_auc, cal_prauc, cal_llloss_with_prob
from data import get_criteo_dataset_stream
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

physical_devices = tf.config.list_physical_devices('GPU')

def test(model, test_data, params):
    all_logits = []
    all_probs = []
    all_labels = []
    start_time = time.time()
    
    for step, (batch_x, batch_y) in enumerate(tqdm(test_data), 1):
        logits = model.predict(batch_x)
        all_logits.append(logits.numpy())
        all_labels.append(batch_y.numpy())
        all_probs.append(tf.sigmoid(logits))
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1,))
    all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1,))
    all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1,))
    if params["method"] == "FNC":
        all_probs = all_probs / (1-all_probs+1e-8)
        llloss = cal_llloss_with_prob(all_labels, all_probs)
    else:
        llloss = cal_llloss_with_logits(all_labels, all_logits)
    batch_size = all_logits.shape[0]
    pred = all_probs >= 0.5
    auc = cal_auc(all_labels, all_probs)
    prauc = cal_prauc(all_labels, all_probs)
    return auc, prauc, llloss, elapsed_ms


def test_our(model, test_data, params):
    all_probs = []
    inw_probs = []
    outw_probs = []
    all_labels = []
    inw_labels = []
    outw_labels = []
    start_time = time.time()
    
    for step, (batch_x, batch_y, batch_y1, batch_y2) in enumerate(tqdm(test_data), 1):
        pred, inw_pred, outw_pred = model.predict(batch_x)
        all_labels.append(batch_y.numpy())
        outw_labels.append(batch_y1.numpy())
        inw_labels.append(batch_y.numpy())
        all_probs.append(pred.numpy())
        inw_probs.append(inw_pred.numpy())
        outw_probs.append(outw_pred.numpy())
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1,))
    all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1,))
    inw_labels = np.reshape(np.concatenate(inw_labels, axis=0), (-1,))
    inw_probs = np.reshape(np.concatenate(inw_probs, axis=0), (-1,))
    outw_labels = np.reshape(np.concatenate(outw_labels, axis=0), (-1,))
    outw_probs = np.reshape(np.concatenate(outw_probs, axis=0), (-1,))
    auc = cal_auc(all_labels, all_probs)
    inw_auc = cal_auc(inw_labels, inw_probs)
    outw_auc = cal_auc(outw_labels, outw_probs)
    prauc = cal_prauc(all_labels, all_probs)
    return auc, inw_auc, outw_auc, prauc, 0, elapsed_ms


def train_our(models, optimizer, train_data, params):
    if params["loss"] == "none_loss":
        return 0
    loss_fn = get_loss_fn(params["loss"])
    start_time = time.time()
    
    for step, batch in enumerate(tqdm(train_data), 1):
        batch_x = batch[0]
        batch_y1 = batch[1]
        batch_y2 = batch[2]
        batch_y3 = batch[3]
        targets = {"label": batch_y1, "outw_label": batch_y2, "inw_mask": batch_y3}
        with tf.GradientTape() as g:
            outputs = models["model"](batch_x, training=True)
            reg_loss = tf.add_n(models["model"].losses)
            loss_dict = loss_fn(targets, outputs, params)
            loss = loss_dict["loss"] + reg_loss

        trainable_variables = models["model"].trainable_variables
        gradients = g.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
    
    elapsed_ms = (time.time() - start_time) * 1000
    return elapsed_ms


def train(models, optimizer, train_data, params):
    if params["loss"] == "none_loss":
        return 0
    loss_fn = get_loss_fn(params["loss"])
    start_time = time.time()
    
    for step, batch in enumerate(tqdm(train_data), 1):
        batch_x = batch[0]
        batch_y1 = batch[1]
        batch_y2 = batch[2]
        batch_y3 = batch[3]
        targets = {"label": batch_y1, "delay_label": batch_y2, "inw_label": batch_y3}

        with tf.GradientTape() as g:
            outputs = models["model"](batch_x, training=True)
            if params["method"] in ["ES-DFM", "DEFUSE", "ES-DFM_1d", "DEFUSE_1d", "DEFUSE_3d", "DEFUSE_7d", "DEFUSE_14d"]:
                logitsx = models["esdfm"](batch_x, training=False)
                outputs = {
                    "logits": outputs["logits"],
                    "tn_logits": logitsx["tn_logits"],
                    "dp_logits": logitsx["dp_logits"]
                }
            elif params["method"] == "DEFER":
                logitsx = models["defer"](batch_x, training=False)
                outputs = {
                    "logits": outputs["logits"],
                    "dp_logits": logitsx["logits"]
                }
            elif params["method"] == "DEFER_unbiased":
                logitsx = models["defer"](batch_x, training=False)
                logitsx2 = models["esdfm"](batch_x, training=False)
                outputs = {
                    "logits": outputs["logits"],
                    "dp_logits": logitsx["logits"],
                    "tn_logits": logitsx2["tn_logits"]
                }
            elif params.get("method") == "AION" and "pre_model" in models:
                pre_out = models["pre_model"](batch_x, training=False)
                try:
                    pre_logits = pre_out["logits"]
                except Exception:
                    pre_logits = pre_out
                if isinstance(outputs, dict):
                    outputs["pre_logits"] = pre_logits
                else:
                    outputs = {"logits": outputs, "pre_logits": pre_logits}
            reg_loss = tf.add_n(models["model"].losses)
            loss_dict = loss_fn(targets, outputs, params)
            loss = loss_dict["loss"] + reg_loss

        trainable_variables = models["model"].trainable_variables
        gradients = g.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
    
    elapsed_ms = (time.time() - start_time) * 1000
    return elapsed_ms


def stream_run_our(params):
    train_stream, test_stream = get_criteo_dataset_stream(params)
    models = {}
    if params["method"] in ["Bi-DEFUSE", "Bi-DEFUSE_1d"]:
        bidefuse_model = get_model("Bi-DEFUSE_inoutw", params)
        bidefuse_model.load_weights(params["pretrain_defuse_model_ckpt_path"])
        models["model"] = bidefuse_model
    elif params["method"] in ["Bi-DEFUSE_MLP", "Bi-DEFUSE_1d_MLP"]:
        bidefuse_model = get_model("Bi-DEFUSE_MLP", params)
        bidefuse_model.load_weights(params["pretrain_defuse_model_ckpt_path"])
        models["model"] = bidefuse_model
    else:
        model = get_model("MLP_SIG", params)
        model.load_weights(params["pretrain_baseline_model_ckpt_path"])
        models = {"model": model}

    if params["method"] == "DEFUSE_inoutw_ind":
        outwmodel = get_model("MLP_SIG", params)
        outwmodel.load_weights(params["pretrain_ours_model_ckpt_path"])
        models["outwmodel"] = outwmodel 

    optimizer = get_optimizer(params["optimizer"], params)

    auc_ma = ScalarMovingAverage()
    auc_inw_ma = ScalarMovingAverage()
    auc_outw_ma = ScalarMovingAverage()
    nll_ma = ScalarMovingAverage()
    prauc_ma = ScalarMovingAverage()
    
    train_time_ma = ScalarMovingAverage()
    test_time_ma = ScalarMovingAverage()
    total_time_ma = ScalarMovingAverage()

    for ep, (train_dataset, test_dataset) in enumerate(zip(train_stream, test_stream)):
        epoch_start_time = time.time()
        
        train_data = tf.data.Dataset.from_tensor_slices(
            (dict(train_dataset["x"]), train_dataset["labels"], 
                train_dataset["delay_labels"], train_dataset["inw_labels"]))
        train_data = train_data.batch(params["batch_size"]).prefetch(1)
        
        train_time_ms = train_our(models, optimizer, train_data, params)

        test_batch_size = test_dataset["x"].shape[0]
        test_data = tf.data.Dataset.from_tensor_slices(
            (dict(test_dataset["x"]), test_dataset["labels"], 
                test_dataset["delay_labels"], test_dataset["inw_labels"]))
        test_data = test_data.batch(params["batch_size"]).prefetch(1)
        
        auc, auc_inw, auc_outw, prauc, llloss, test_time_ms = test_our(models["model"], test_data, params)
        
        epoch_total_ms = (time.time() - epoch_start_time) * 1000
        
        train_time_ma.add(train_time_ms, 1)
        test_time_ma.add(test_time_ms, 1)
        total_time_ma.add(epoch_total_ms, 1)
        
        print(f"epoch {ep}, train_time: {train_time_ms:.2f} ms, test_time: {test_time_ms:.2f} ms, total_time: {epoch_total_ms:.2f} ms")
        print(f"epoch {ep}, auc {auc:.6f}, auc_inw {auc_inw:.6f}, auc_outw {auc_outw:.6f}, prauc {prauc:.6f}, llloss {llloss:.6f}")
        
        auc_ma.add(auc*test_batch_size, test_batch_size)
        auc_inw_ma.add(auc_inw*test_batch_size, test_batch_size)
        auc_outw_ma.add(auc_outw*test_batch_size, test_batch_size)
        nll_ma.add(llloss*test_batch_size, test_batch_size)
        prauc_ma.add(prauc*test_batch_size, test_batch_size)
        
        print(f"epoch {ep}, auc_ma {auc_ma.get():.6f}, auc_inw_ma {auc_inw_ma.get():.6f}, auc_outw_ma {auc_outw_ma.get():.6f}, prauc_ma {prauc_ma.get():.6f}, llloss_ma {nll_ma.get():.6f}")
        print(f"AVG TIME - train: {train_time_ma.get():.2f} ms, test: {test_time_ma.get():.2f} ms, total: {total_time_ma.get():.2f} ms")

    return

def stream_run(params):
    train_stream, test_stream = get_criteo_dataset_stream(params)
    if params["method"] == "DFM":
        model = get_model("MLP_EXP_DELAY", params)
        model.load_weights(params["pretrain_dfm_model_ckpt_path"])
    elif params["method"] == "AION":
        model = get_model("MLP_SIG", params)
        model.load_weights(params["pretrain_baseline_model_ckpt_path"])
        pre_model = get_model("MLP_SIG", params)
        pre_model.load_weights(params["pretrain_baseline_model_ckpt_path"])
    else:
        model = get_model("MLP_SIG", params)
        model.load_weights(params["pretrain_baseline_model_ckpt_path"])
    models = {"model": model}
    if params["method"] == "AION":
        try:
            models["pre_model"] = pre_model
        except NameError:
            pass
    if params["method"] in ["ES-DFM", "DEFUSE", "ES-DFM_1d", "DEFUSE_1d", "DEFUSE_3d", "DEFUSE_7d", "DEFUSE_14d"]:
        esdfm_model = get_model("MLP_tn_dp", params)
        esdfm_model.load_weights(params["pretrain_esdfm_model_ckpt_path"])
        models["esdfm"] = esdfm_model
    elif params["method"] in ["DEFER", "DEFER_unbiased"]:
        defer_model = get_model("MLP_dp", params)
        defer_model.load_weights(params["pretrain_defer_model_ckpt_path"])
        models["defer"] = defer_model
        if params["method"] == "DEFER_unbiased":
            esdfm_model = get_model("MLP_tn_dp", params)
            esdfm_model.load_weights(params["pretrain_esdfm_model_ckpt_path"])
            models["esdfm"] = esdfm_model
    elif params["method"] == "DFM":
        dfm_model = get_model("MLP_EXP_DELAY", params)
        dfm_model.load_weights(params["pretrain_dfm_model_ckpt_path"])
        models["model"] = dfm_model

    optimizer = get_optimizer(params["optimizer"], params)

    auc_ma = ScalarMovingAverage()
    nll_ma = ScalarMovingAverage()
    prauc_ma = ScalarMovingAverage()
    
    train_time_ma = ScalarMovingAverage()
    test_time_ma = ScalarMovingAverage()
    total_time_ma = ScalarMovingAverage()

    for ep, (train_dataset, test_dataset) in enumerate(zip(train_stream, test_stream)):
        epoch_start_time = time.time()  
        

        if params["method"] in ['FNW_1d', 'FNC_1d']:
            train_data = tf.data.Dataset.from_tensor_slices(
                (dict(train_dataset["x"]), train_dataset["labels"]))
        else:
            train_data = tf.data.Dataset.from_tensor_slices(
                (dict(train_dataset["x"]), train_dataset["labels"], 
                    train_dataset["delay_labels"], train_dataset["inw_labels"]))
        train_data = train_data.batch(params["batch_size"]).prefetch(1)
        

        train_time_ms = train(models, optimizer, train_data, params)

  
        test_batch_size = test_dataset["x"].shape[0]
        test_data = tf.data.Dataset.from_tensor_slices(
            (dict(test_dataset["x"]), test_dataset["labels"]))
        test_data = test_data.batch(params["batch_size"]).prefetch(1)
        

        auc, prauc, llloss, test_time_ms = test(model, test_data, params)
        
        epoch_total_ms = (time.time() - epoch_start_time) * 1000  # 整个epoch总时间
        

        train_time_ma.add(train_time_ms, 1)
        test_time_ma.add(test_time_ms, 1)
        total_time_ma.add(epoch_total_ms, 1)
        
        print(f"epoch {ep}, train_time: {train_time_ms:.2f} ms, test_time: {test_time_ms:.2f} ms, total_time: {epoch_total_ms:.2f} ms")
        print(f"epoch {ep}, auc {auc:.6f}, prauc {prauc:.6f}, llloss {llloss:.6f}")
        
        auc_ma.add(auc*test_batch_size, test_batch_size)
        nll_ma.add(llloss*test_batch_size, test_batch_size)
        prauc_ma.add(prauc*test_batch_size, test_batch_size)
        
        print(f"epoch {ep}, auc_ma {auc_ma.get():.6f}, prauc_ma {prauc_ma.get():.6f}, llloss_ma {nll_ma.get():.6f}")
        print(f"AVG TIME - train: {train_time_ma.get():.2f} ms, test: {test_time_ma.get():.2f} ms, total: {total_time_ma.get():.2f} ms")

    return