import os, re, glob, argparse, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

def _natural_key(s): return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e / (e.sum(axis=axis, keepdims=True)+1e-12)

def load_base_adj(adjdata):
    import pickle
    with open(adjdata, 'rb') as f:
        obj = pickle.load(f, encoding='latin1')
    if isinstance(obj, (list, tuple)) and len(obj)==3:
        sensor_ids, sensor_id_to_ind, adj_mx = obj
    else:
        adj_mx = obj
    A = adj_mx[0] if isinstance(adj_mx, (list,tuple)) else adj_mx
    return np.asarray(A, dtype=np.float32)

def load_all_series(data_dir):
    def pick(x):  # x: [B,T,N,C] -> [BT,N] (C=0)
        B,T,N,C = x.shape
        return x.reshape(B*T, N, C)[:,:,0]
    xs=[]
    for sp in ["train","val","test"]:
        p=os.path.join(data_dir, f"{sp}.npz")
        if os.path.exists(p):
            d=np.load(p)
            xs.append(pick(d['x']))
    if not xs: raise FileNotFoundError(f"not found {data_dir}/(train|val|test).npz")
    return np.concatenate(xs, axis=0)  # [T,N]

def corr_matrix(X_TN):
    X = X_TN - X_TN.mean(0, keepdims=True)
    std = X.std(0, keepdims=True) + 1e-8
    Z = X / std
    return (Z.T @ Z) / (Z.shape[0]-1)

def find_checkpoints(save_dir):
    cks = glob.glob(os.path.join(save_dir, "*.pth"))
    epoch_ckpts, best_ckpt = [], None
    for p in cks:
        if "_epoch_" in os.path.basename(p): epoch_ckpts.append(p)
        if "_best_" in os.path.basename(p) and best_ckpt is None: best_ckpt = p
    epoch_ckpts.sort(key=_natural_key)
    return epoch_ckpts, best_ckpt

def extract_adaptive_A(sd):
    # 兼容 nodevec1/nodevec2/embed1/embed2
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    keys = list(sd.keys())
    def pick(*cands):
        for k in cands:
            if k in sd: return sd[k]
        return None
    E1 = pick('nodevec1','embed1','E1','module.nodevec1','graph.nodevec1')
    E2 = pick('nodevec2','embed2','E2','module.nodevec2','graph.nodevec2')
    if E1 is None and E2 is None:
        E = pick('nodevec','embed','E','module.nodevec')
        if E is None: return None
        E = E.detach().cpu().numpy()
        A = _softmax(np.maximum(E@E.T, 0.0), axis=1)
        return A.astype(np.float32)
    E1 = E1.detach().cpu().numpy()
    E2 = (E1 if E2 is None else E2.detach().cpu().numpy())
    A = _softmax(np.maximum(E1@E2.T, 0.0), axis=1)
    return A.astype(np.float32)

def plot_heatmap(M, title, out):
    plt.figure(figsize=(5,4)); plt.imshow(M, aspect='auto'); plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title); plt.tight_layout(); plt.savefig(out, dpi=180); plt.close()

def plot_series(xs, ys_dict, out, xlabel="epoch"):
    from matplotlib.ticker import MaxNLocator
    plt.figure(figsize=(6,4))
    for k,v in ys_dict.items(): plt.plot(xs, v, label=k, linewidth=2)
    plt.xlabel(xlabel); plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=180); plt.close()

def topk_mask(A, k=3):
    A2 = np.zeros_like(A); idx = np.argpartition(-A, kth=min(k-1, A.shape[1]-1), axis=1)[:, :k]
    rows = np.arange(A.shape[0])[:, None]; A2[rows, idx] = A[rows, idx]; return A2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--adjdata", required=True)
    ap.add_argument("--project", default="")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--log_to_wandb", action="store_true")
    ap.add_argument("--k_top", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    figs_dir = os.path.join(args.save_dir, "viz"); os.makedirs(figs_dir, exist_ok=True)

    # Base adjacency & correlation
    A_base = load_base_adj(args.adjdata)
    plot_heatmap(A_base, "Base adjacency", os.path.join(figs_dir, "A_base.png"))

    X_TN = load_all_series(args.data)  # [T,N]
    C = corr_matrix(X_TN)
    plot_heatmap(C, "Correlation (Pearson)", os.path.join(figs_dir, "Corr_pearson.png"))

    # Epoch checkpoints
    epochs, diag_m, offdiag_m, asym_m, specr_m, simC_m = [], [], [], [], [], []
    epoch_ckpts, best_ckpt = find_checkpoints(args.save_dir)

    for pth in epoch_ckpts:
        m = re.search(r"_epoch_(\d+)_", os.path.basename(pth)); e = int(m.group(1)) if m else len(epochs)+1
        sd = torch.load(pth, map_location="cpu")
        A = extract_adaptive_A(sd)
        if A is None: continue
        # 快照图
        if e in [1,2,3] or (len(epoch_ckpts)>=5 and e % max(1, len(epoch_ckpts)//5)==0) or (pth==epoch_ckpts[-1]):
            plot_heatmap(A, f"A_adp (epoch {e})", os.path.join(figs_dir, f"A_adp_epoch{e}.png"))
            plot_heatmap(topk_mask(A, k=args.k_top), f"A_adp top-{args.k_top} (epoch {e})", os.path.join(figs_dir, f"A_adp_top{args.k_top}_epoch{e}.png"))
        # 统计
        diag_m.append(float(np.diag(A).mean()))
        offdiag_m.append(float((A.sum()-np.trace(A))/(A.size - A.shape[0])))
        asym_m.append(float(np.linalg.norm(A-A.T,'fro')/(np.linalg.norm(A,'fro')+1e-8)))
        specr_m.append(float(np.max(np.abs(np.linalg.eigvals(A)))))
        iu = np.triu_indices_from(A, k=1)
        from scipy.stats import spearmanr
        rho, _ = spearmanr(A[iu].ravel(), C[iu].ravel())
        simC_m.append(float(rho))
        epochs.append(e)

    if epochs:
        plot_series(epochs, {"diag_mean":diag_m, "offdiag_mean":offdiag_m, "asym_fro":asym_m}, os.path.join(figs_dir,"evolution_stat1.png"))
        plot_series(epochs, {"spectral_radius":specr_m, "spearman(A,C)":simC_m}, os.path.join(figs_dir,"evolution_stat2.png"))

    if best_ckpt:
        sd = torch.load(best_ckpt, map_location="cpu")
        A_best = extract_adaptive_A(sd)
        if A_best is not None:
            plot_heatmap(A_best, "A_adp (best)", os.path.join(figs_dir, "A_adp_best.png"))
            Cn = (C - C.min())/(C.max()-C.min()+1e-12)
            plot_heatmap(A_best - Cn, "A_best - Corr(norm)", os.path.join(figs_dir, "A_best_minus_corr.png"))

    # wandb 上传（*）
    if args.log_to_wandb and args.project and args.run_name:
        import wandb
        run = wandb.init(project=args.project, name=args.run_name, resume="allow")
        for fn in sorted(glob.glob(os.path.join(figs_dir, "*.png"))):
            run.log({"viz/"+os.path.basename(fn): wandb.Image(fn)})
        run.finish()

if __name__ == "__main__":
    main()
