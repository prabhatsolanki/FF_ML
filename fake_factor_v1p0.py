#!/usr/bin/env python

"""
Fake Factor Estimation using -
  - A Neural Network classifier (PyTorch) with p/(1-p) reweighting.
  - A Normalizing Flow reweighter (using rational quadratic spline coupling layers with permutation).
  - A benchmark XGBoost reweighter (BDT).
  - Ani terative BDT reweighter that uses a custom decision tree with a symmetrized χ² splitting criterion (As done in SUS-23-007).
  
In test mode the models and synthetic data are loaded and evaluation (closure plots) is performed.
  
Usage:
    python fake_factor_v1p0.py --mode train --synthetic True --N 500000
    python fake_factor_v1p0.py --mode test --synthetic True
If --synthetic is False, real data should be provided (currently returns None).
"""

import os, sys, argparse, logging, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("training.log")])
logger = logging.getLogger(__name__)

np.random.seed(42)

# =========================================
# Data Generation and Preprocessing Functions
# =========================================
def generate_synthetic_data(N=500000):
    logger.info("Generating synthetic data...")
    tau_pt = 20 + np.random.exponential(50, size=N)  
    tau_eta = np.random.uniform(-2.4, 2.4, size=N)
    tau_charge = np.random.choice([-1, 1], size=N)
    other_charge = np.random.choice([-1, 1], size=N)
    total_charge = tau_charge + other_charge
    a, b = (0.5 - 1.0) / 0.5, (2.0 - 1.0) / 0.5
    jet_pt_ratio = truncnorm.rvs(a, b, loc=1.0, scale=0.5, size=N)
    Njets = np.random.choice([0,1,2,3,4], size=N, p=[0.1, 0.5, 0.2, 0.15, 0.05])
    years = [2016, 2017, 2018]
    year = np.random.choice(years, size=N)
    decay_modes = [0, 1, 10]
    decayMode = np.random.choice(decay_modes, size=N, p=[0.4, 0.4, 0.2])
    lin_score = (-1.0 + 0.02*tau_pt - 0.5*np.abs(tau_eta) - 3.0*(jet_pt_ratio-1.0)
                 - 0.3*Njets - 0.5*(decayMode==10).astype(float))
    base_prob = 1/(1+np.exp(-lin_score))
    a_param = base_prob * 10
    b_param = (1-base_prob) * 10
    ID_score = np.random.beta(a_param, b_param)
    label = (ID_score >= 0.75).astype(int)
    data = pd.DataFrame({
        'tau_pt': tau_pt,
        'tau_eta': tau_eta,
        'tau_charge': tau_charge,
        'total_charge': total_charge,
        'jet_pt_ratio': jet_pt_ratio,
        'Njets': Njets,
        'year': year,
        'decayMode': decayMode,
        'ID_score': ID_score,
        'label': label
    })
    logger.info("Synthetic data generation complete.")
    return data

def preprocess_data(data):
    logger.info("Preprocessing data...")
    features_df = data.drop(columns=['ID_score', 'label'])
    features_df = pd.get_dummies(features_df, columns=['year','decayMode'], prefix=['year','decay'])
    features_df['tau_pt'] = features_df['tau_pt'] / 100.0  # scale tau_pt
    X = features_df.values.astype('float32')
    y = data['label'].values.astype('float32')
    return X, y

# =========================================
# NN Classifier Reweighter
# =========================================
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class FFClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FFClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_nn_classifier(X_train, y_train, X_val, y_val, batch_size=10000, num_epochs=50, patience=5, lr=1e-2):
    logger.info("Training NN classifier...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    model = FFClassifier(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).view(-1,1))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).view(-1,1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered for NN.")
                break
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), "ff_classifier_nn.pth")
    logger.info("NN classifier model saved as ff_classifier_nn.pth.")
    return model

def test_nn_classifier(model, X_test, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_t = torch.from_numpy(X_test).to(device)
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t)
        probs_test = torch.sigmoid(logits_test).cpu().numpy().flatten()
    fail_mask = (y_test == 0)
    weights = probs_test[fail_mask] / (1 - probs_test[fail_mask])
    return weights

# =========================================
# NF Reweighter (Spline Coupling Only)
# =========================================
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
def fixed_permutation(input_dim, seed, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randperm(input_dim, generator=gen, device=device)
def create_nf(input_dim, num_layers=12, hidden_dims_spline=[256,256], count_bins=16, bound=8.0, device=torch.device("cuda")):
    transforms = []
    for i in range(num_layers):
        transforms.append(T.spline_coupling(input_dim=input_dim,
                                            count_bins=count_bins,
                                            hidden_dims=hidden_dims_spline,
                                            bound=bound))
        perm = fixed_permutation(input_dim, seed=42+i, device=device)
        transforms.append(T.Permute(perm))
    flow_transform = T.ComposeTransformModule(transforms)
    base_dist = dist.Normal(torch.zeros(input_dim, device=device),
                              torch.ones(input_dim, device=device)).to_event(1)
    flow = dist.TransformedDistribution(base_dist, [flow_transform])
    return flow, flow_transform

def train_nf(flow, flow_transform, optimizer, data_tensor, num_steps=500, batch_size=10000):
    flow_transform.train()
    num_batches = int(np.ceil(data_tensor.size(0) / batch_size))
    for step in range(num_steps):
        perm = torch.randperm(data_tensor.size(0))
        loss_batch = 0
        for i in range(0, data_tensor.size(0), batch_size):
            idx = perm[i:i+batch_size]
            batch = data_tensor[idx]
            nll = -flow.log_prob(batch).mean()
            optimizer.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(flow_transform.parameters(), max_norm=5.0)
            optimizer.step()
            loss_batch += nll.item()
        if step % 10 == 0:
            logger.info(f"NF Step {step}: Avg NLL = {loss_batch/num_batches:.3f}")
    return flow

def test_nf(flow_fail, flow_pass, X_test_fail, device=torch.device("cuda")):
    with torch.no_grad():
        log_prob_fail = flow_fail.log_prob(X_test_fail)
        log_prob_pass = flow_pass.log_prob(X_test_fail)
    weights = torch.exp(log_prob_pass - log_prob_fail).cpu().numpy().flatten()
    return weights

# =========================================
# XGBoost BDT Reweighter
# =========================================
import xgboost as xgb

def train_xgb_reweighter(X_train, y_train, X_val, y_val, params=None, num_boost_round=1000, early_stopping_rounds=20):
    logger.info("Training XGBoost reweighter...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 20,
            'eta': 0.05,
            'tree_method': 'hist',
            'device': 'cuda'
        }
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dval, "val")],
                    early_stopping_rounds=early_stopping_rounds, verbose_eval=10)
    bst.save_model("ff_xgb_model.json")
    logger.info("XGBoost model saved as ff_xgb_model.json.")
    return bst

def test_xgb_reweighter(bst, X_test, y_test):
    dtest = xgb.DMatrix(X_test, label=y_test)
    probs = bst.predict(dtest)
    fail_mask = (y_test == 0)
    weights = probs[fail_mask] / (1 - probs[fail_mask])
    return weights

##############################
# Iterative BDT Reweighter with Symmetrized χ² Splitting 
# (To be replaced by the more optimsed implementation as implemented in SUS-23-007)
##############################
from sklearn.base import clone

class CustomTreeNode:
    def __init__(self, depth=0):
        self.depth = depth
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.log_ratio = None

def chi2_metric(w_mc, w_rd, eps=1e-6):
    w_mc = np.clip(w_mc, 1e-8, 1e8)
    w_rd = np.clip(w_rd, 1e-8, 1e8)
    return (w_mc - w_rd)**2 / (w_mc + w_rd + eps)

def build_custom_tree(X, y, weights, depth=0, max_depth=2, min_samples=50, eps=1e-6):
    """
    Recursively build a custom decision tree using the symmetrized chi² metric.
    """
    node = CustomTreeNode(depth=depth)
    n_samples = X.shape[0]
    
    if n_samples < min_samples or depth >= max_depth:
        # At leaf, set correction: log((sum_{MC}+eps)/(sum_{RD}+eps))
        w_mc = weights[y==0].sum()
        w_rd = weights[y==1].sum()
        node.is_leaf = True
        node.log_ratio = np.log((w_mc + eps) / (w_rd + eps))
        return node
    
    best_chi2 = -np.inf
    best_feature, best_threshold = None, None
    best_left_idx, best_right_idx = None, None
    
    for j in range(X.shape[1]):
        vals = X[:, j]
        unique_vals = np.unique(vals)
        if len(unique_vals) < 2:
            continue
        thresholds = np.percentile(vals, np.linspace(10, 90, 10))
        for thresh in thresholds:
            left_idx = np.where(vals <= thresh)[0]
            right_idx = np.where(vals > thresh)[0]
            if len(left_idx) < min_samples or len(right_idx) < min_samples:
                continue
            w_mc_left = weights[left_idx][y[left_idx]==0].sum()
            w_rd_left = weights[left_idx][y[left_idx]==1].sum()
            w_mc_right = weights[right_idx][y[right_idx]==0].sum()
            w_rd_right = weights[right_idx][y[right_idx]==1].sum()
            chi2_left = chi2_metric(w_mc_left, w_rd_left, eps)
            chi2_right = chi2_metric(w_mc_right, w_rd_right, eps)
            chi2_total = chi2_left + chi2_right
            if chi2_total > best_chi2:
                best_chi2 = chi2_total
                best_feature = j
                best_threshold = thresh
                best_left_idx = left_idx
                best_right_idx = right_idx
    if best_feature is None:
        w_mc = weights[y==0].sum()
        w_rd = weights[y==1].sum()
        node.is_leaf = True
        node.log_ratio = np.log((w_mc + eps) / (w_rd + eps))
        return node
    node.feature = best_feature
    node.threshold = best_threshold
    node.left = build_custom_tree(X[best_left_idx], y[best_left_idx], weights[best_left_idx],
                                  depth=depth+1, max_depth=max_depth, min_samples=min_samples, eps=eps)
    node.right = build_custom_tree(X[best_right_idx], y[best_right_idx], weights[best_right_idx],
                                   depth=depth+1, max_depth=max_depth, min_samples=min_samples, eps=eps)
    return node

def predict_tree(node, x, eps=1e-6):
    if node.is_leaf:
        return node.log_ratio
    if x[node.feature] <= node.threshold:
        return predict_tree(node.left, x, eps)
    else:
        return predict_tree(node.right, x, eps)

def train_iterative_custom_bdt(df, features, label_col="label", n_iterations=10, 
                               max_depth=2, min_samples=50, eps=1e-6, clip_val=5):

    n = len(df)
    log_weights = np.zeros(n)
    y_vals = df[label_col].values.astype(int)
    trees = []
    for it in range(n_iterations):
        print(f"Iteration:{it}")
        X = df[features].values
        current_weights = np.ones(n)
        current_weights[y_vals==0] = np.exp(log_weights[y_vals==0])
        tree = build_custom_tree(X, y_vals, current_weights, max_depth=max_depth, min_samples=min_samples, eps=eps)
        for i in range(n):
            if y_vals[i] == 0:
                update = predict_tree(tree, X[i])
                update = np.clip(update, -clip_val, clip_val)
                log_weights[i] += update
        trees.append(tree)
    final_weights = np.ones(n)
    log_weights_clipped = np.clip(log_weights, -50, 50)
    final_weights[y_vals==0] = np.exp(log_weights_clipped[y_vals==0])
    return trees, final_weights

def apply_iterative_custom_bdt(trees, X, y, eps=1e-6, clip_val=5):
    n = X.shape[0]
    log_weights = np.zeros(n)
    for tree in trees:
        for i in range(n):
            if y[i] == 0:
                update = predict_tree(tree, X[i])
                log_weights[i] += np.clip(update, -clip_val, clip_val)
    return np.exp(log_weights)

def train_and_apply_custom_bdt(train_df, test_df, features, label_col="label", n_iterations=10,
                               max_depth=2, min_samples=50, clip_val=5):
    trees, _ = train_iterative_custom_bdt(train_df, features, label_col, n_iterations, max_depth, min_samples, clip_val=clip_val)
    X_test_custom = test_df[features].values.astype('float32')
    y_test_custom = test_df[label_col].values.astype(int)
    weights = apply_iterative_custom_bdt(trees, X_test_custom, y_test_custom, clip_val=clip_val)
    return weights

# =========================================
# Closure Tests Function 
# =========================================
def run_closure_tests(X_test, y_test, df_test_custom, nn_weights, nf_weights, xgb_weights, custom_weights):
    # columns: 0: tau_pt, 1: tau_eta, 3: jet_pt_ratio, 4: Njets.
    fail_mask = (y_test == 0)
    tau_pt_test_fail = X_test[fail_mask][:, 0] * 100.0
    tau_pt_test_pass = X_test[~fail_mask][:, 0] * 100.0
    tau_eta_test_fail = X_test[fail_mask][:, 1]
    tau_eta_test_pass = X_test[~fail_mask][:, 1]
    jet_pt_ratio_test_fail = X_test[fail_mask][:, 3]
    jet_pt_ratio_test_pass = X_test[~fail_mask][:, 3]
    Njets_test_fail = X_test[fail_mask][:, 4]
    Njets_test_pass = X_test[~fail_mask][:, 4]
    
    bins_pt = np.linspace(20, 250, 50)
    bins_eta = np.linspace(-2.4, 2.4, 25)
    bins_ratio = np.linspace(0.1, 2.5, 25)
    bins_njets = np.arange(-0.5, 5.5, 1)
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))
    w_custom_test = custom_weights[np.array(df_test_custom["label"])==0]
    
    def plot_closure(ax, pass_data, fail_data, w_nn, w_nf, w_xgb, w_custom, title, xlabel, bins):
        ax.hist(pass_data, bins=bins, density=True, histtype='step', linewidth=2, label='Pass (Truth)')
        ax.hist(fail_data, bins=bins, density=True, histtype='step', linestyle='dashed', linewidth=2, label='Fail (Raw)')
        ax.hist(fail_data, bins=bins, weights=w_nn, density=True, histtype='step', linewidth=2, label='Fail (NN-weighted)')
        ax.hist(fail_data, bins=bins, weights=w_nf, density=True, histtype='step', linewidth=2, label='Fail (NF-weighted)')
        ax.hist(fail_data, bins=bins, weights=w_xgb, density=True, histtype='step', linewidth=2, label='Fail (XGB-weighted)')
        ax.hist(fail_data, bins=bins, weights=w_custom, density=True, histtype='step', linewidth=2, label='Fail (Custom BDT)')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend()
    
    plot_closure(axs[0], tau_pt_test_pass, tau_pt_test_fail, nn_weights, nf_weights, xgb_weights, w_custom_test,
                 "Tau $p_T$ Distribution", "Tau $p_T$ [GeV]", bins_pt)
    plot_closure(axs[1], tau_eta_test_pass, tau_eta_test_fail, nn_weights, nf_weights, xgb_weights, w_custom_test,
                 "Tau $\eta$ Distribution", "Tau $\eta$", bins_eta)
    #plot_closure(axs[1,0], jet_pt_ratio_test_pass, jet_pt_ratio_test_fail, nn_weights, nf_weights, xgb_weights, w_custom_test,
    #             "Jet $p_T$/Tau $p_T$ Ratio", "Jet $p_T$/Tau $p_T$", bins_ratio)
    plot_closure(axs[2], Njets_test_pass, Njets_test_fail, nn_weights, nf_weights, xgb_weights, w_custom_test,
                 "Number of Jets", "Njets", bins_njets)
    plt.tight_layout()
    plt.savefig("closure_plots_all.png", dpi=300)
    plt.show()
    logger.info("Closure plots saved as 'closure_plots_all.png'.")

# =========================================
# Main Function
# =========================================
def main(args):
    if not args.synthetic:
        logger.info("Synthetic flag is False. Real data not provided. Exiting.")
        return None
    
    if args.mode == "train":
        data = generate_synthetic_data(N=args.N)
        with open("synthetic_data.pkl", "wb") as f:
            pickle.dump(data, f)
        logger.info("Synthetic data saved as synthetic_data.pkl.")
    else:
        with open("synthetic_data.pkl", "rb") as f:
            data = pickle.load(f)
        logger.info("Loaded synthetic data from synthetic_data.pkl.")
    
    X, y = preprocess_data(data)
    train_val_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)
    X_train = X[np.array(train_idx)]
    y_train = y[np.array(train_idx)]
    X_val = X[np.array(val_idx)]
    y_val = y[np.array(val_idx)]
    X_test = X[np.array(test_idx)]
    y_test = y[np.array(test_idx)]
    logger.info(f"Data split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    
    global tau_pt_full, tau_eta_full, jet_pt_ratio_full, Njets_full
    tau_pt_full = data['tau_pt'].values  
    tau_eta_full = data['tau_eta'].values
    jet_pt_ratio_full = data['jet_pt_ratio'].values
    Njets_full = data['Njets'].values
    
    # NN Classifier
    if args.mode == "train":
        model_nn = train_nn_classifier(X_train, y_train, X_val, y_val, batch_size=args.batch_size_nn,
                                       num_epochs=args.epochs_nn, patience=args.patience_nn, lr=args.lr_nn)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = X_train.shape[1]
        model_nn = FFClassifier(input_dim).to(device)
        model_nn.load_state_dict(torch.load("ff_classifier_nn.pth", map_location=device))
        model_nn.eval()
        logger.info("Loaded NN classifier model.")
    nn_weights = test_nn_classifier(model_nn, X_test, y_test)
    
    # NF Reweighter
    flow_indices = args.flow_indices.split(",")
    flow_indices = [int(i) for i in flow_indices]
    X_flow = X[:, flow_indices]
    fail_idx_all = (y == 0)
    pass_idx_all = (y == 1)
    X_fail = X_flow[fail_idx_all]
    X_pass = X_flow[pass_idx_all]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_fail_t = torch.tensor(X_fail, dtype=torch.float32, device=device)
    X_pass_t = torch.tensor(X_pass, dtype=torch.float32, device=device)
    
    if args.mode == "train":
        nf_fail, nf_fail_transform = create_nf(input_dim=len(flow_indices),
                                               num_layers=args.nf_layers,
                                               hidden_dims_spline=[args.nf_hidden]*2,
                                               count_bins=args.nf_bins,
                                               bound=args.nf_bound,
                                               device=device)
        nf_pass, nf_pass_transform = create_nf(input_dim=len(flow_indices),
                                               num_layers=args.nf_layers,
                                               hidden_dims_spline=[args.nf_hidden]*2,
                                               count_bins=args.nf_bins,
                                               bound=args.nf_bound,
                                               device=device)
        nf_fail_transform.to(device)
        nf_pass_transform.to(device)
        optimizer_fail = torch.optim.Adam(nf_fail_transform.parameters(), lr=args.lr_nf)
        nf_fail = train_nf(nf_fail, nf_fail_transform, optimizer_fail, X_fail_t, num_steps=args.nf_steps, batch_size=args.batch_size_nf)
        optimizer_pass = torch.optim.Adam(nf_pass_transform.parameters(), lr=args.lr_nf)
        nf_pass = train_nf(nf_pass, nf_pass_transform, optimizer_pass, X_pass_t, num_steps=args.nf_steps, batch_size=args.batch_size_nf)
        torch.save(nf_fail_transform.state_dict(), "ff_flow_fail.pth")
        torch.save(nf_pass_transform.state_dict(), "ff_flow_pass.pth")
        logger.info("NF models saved.")
    else:
        nf_fail, nf_fail_transform = create_nf(input_dim=len(flow_indices),
                                               num_layers=args.nf_layers,
                                               hidden_dims_spline=[args.nf_hidden]*2,
                                               count_bins=args.nf_bins,
                                               bound=args.nf_bound,
                                               device=device)
        nf_pass, nf_pass_transform = create_nf(input_dim=len(flow_indices),
                                               num_layers=args.nf_layers,
                                               hidden_dims_spline=[args.nf_hidden]*2,
                                               count_bins=args.nf_bins,
                                               bound=args.nf_bound,
                                               device=device)
        nf_fail_transform.load_state_dict(torch.load("ff_flow_fail.pth", map_location=device))
        nf_pass_transform.load_state_dict(torch.load("ff_flow_pass.pth", map_location=device))
        nf_fail_transform.to(device)
        nf_pass_transform.to(device)
        logger.info("Loaded NF models.")
    X_test_flow = X_test[:, flow_indices]
    X_test_flow_t = torch.tensor(X_test_flow, dtype=torch.float32, device=device)
    fail_mask_flow = (y_test == 0)
    X_test_flow_fail = X_test_flow_t[fail_mask_flow]
    nf_weights = test_nf(nf_fail, nf_pass, X_test_flow_fail, device=torch.device("cuda"))
    
    # XGBoost Reweighter
    if args.mode == "train":
        bst = train_xgb_reweighter(X_train, y_train, X_val, y_val, num_boost_round=args.xgb_rounds, early_stopping_rounds=args.xgb_patience,
                                   params={'objective':'binary:logistic','eval_metric':'logloss','max_depth':args.xgb_max_depth,
                                           'eta':args.xgb_eta,'tree_method':'hist','device':'cuda'})
    else:
        bst = xgb.Booster()
        bst.load_model("ff_xgb_model.json")
        logger.info("Loaded XGBoost model.")
    xgb_weights = test_xgb_reweighter(bst, X_test, y_test)
    
    # Iterative BDT Reweighter
    df_train_custom = data.loc[train_idx, ["tau_pt", "tau_eta","tau_charge","total_charge", "jet_pt_ratio", "Njets","decayMode", "label"]].copy()
    df_test_custom = data.loc[test_idx, ["tau_pt", "tau_eta","tau_charge","total_charge", "jet_pt_ratio", "Njets","decayMode", "label"]].copy()
    features_custom = ["tau_pt", "tau_eta","total_charge", "jet_pt_ratio"]
    if args.mode == "train":
        custom_weights = train_and_apply_custom_bdt(df_train_custom, df_test_custom, features_custom, label_col="label",
                                                    n_iterations=args.custom_iter, max_depth=args.custom_max_depth, 
                                                    min_samples=args.custom_min_samples, clip_val=args.custom_clip)
        with open("ff_custom_bdt.pkl", "wb") as f:
            pickle.dump(custom_weights, f)
        logger.info("Custom iterative BDT weights saved to ff_custom_bdt.pkl.")
    else:
        with open("ff_custom_bdt.pkl", "rb") as f:
            custom_weights = pickle.load(f)
        logger.info("Loaded custom iterative BDT weights.")
    
    # Run Closure Tests
    run_closure_tests(X_test, y_test, df_test_custom, nn_weights, nf_weights, xgb_weights, custom_weights)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML-based Fake Factor Estimation")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Run training or test mode")
    parser.add_argument("--synthetic", type=bool, default=True, help="Use synthetic data (True) or real data (False; returns None)")
    parser.add_argument("--N", type=int, default=50000, help="Number of events for synthetic data")
    # NN parameters
    parser.add_argument("--batch_size_nn", type=int, default=50000, help="Batch size for NN training")
    parser.add_argument("--epochs_nn", type=int, default=100, help="Number of epochs for NN training")
    parser.add_argument("--patience_nn", type=int, default=5, help="Early stopping patience for NN training")
    parser.add_argument("--lr_nn", type=float, default=1e-3, help="Learning rate for NN")
    # NF parameters
    parser.add_argument("--flow_indices", type=str, default="0,1,3,4", help="Comma-separated feature indices for NF")
    parser.add_argument("--nf_layers", type=int, default=8, help="Number of layers for NF")
    parser.add_argument("--nf_hidden", type=int, default=128, help="Hidden layer size for NF spline coupling")
    parser.add_argument("--nf_bins", type=int, default=18, help="Number of bins for NF spline coupling")
    parser.add_argument("--nf_bound", type=float, default=8.0, help="Bound for NF spline coupling")
    parser.add_argument("--lr_nf", type=float, default=1e-3, help="Learning rate for NF")
    parser.add_argument("--nf_steps", type=int, default=500, help="Number of training steps for NF")
    parser.add_argument("--batch_size_nf", type=int, default=50000, help="Batch size for NF training")
    # XGBoost parameters
    parser.add_argument("--xgb_max_depth", type=int, default=10, help="Max depth for XGBoost")
    parser.add_argument("--xgb_eta", type=float, default=0.05, help="Eta for XGBoost")
    parser.add_argument("--xgb_rounds", type=int, default=200, help="Number of boosting rounds for XGBoost")
    parser.add_argument("--xgb_patience", type=int, default=20, help="Early stopping rounds for XGBoost")
    # Custom iterative BDT parameters
    parser.add_argument("--custom_iter", type=int, default=60, help="Number of iterations for custom BDT reweighter")
    parser.add_argument("--custom_max_depth", type=int, default=20, help="Max depth for custom BDT reweighter")
    parser.add_argument("--custom_min_samples", type=int, default=50, help="Min samples per leaf for custom BDT reweighter")
    parser.add_argument("--custom_clip", type=float, default=5, help="Clip value for custom BDT reweighter updates")
    
    args = parser.parse_args()
    main(args)