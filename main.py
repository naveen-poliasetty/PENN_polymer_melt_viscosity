import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
import os
import warnings
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import wandb

from src.models.penn import Visc_PENN, EarlyStopping
from src.data.preprocessing import run_preprocessing_pipeline
from src.data.splitting import generate_extrapolation_splits, get_train_test_dataframes
from src.data.dataloader import get_dataloaders

warnings.filterwarnings('ignore')

def load_and_format_dataset(csv_path, n_fp=1024):
    print("Loading and parsing raw dataset...")
    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw.dropna(subset=['Mw', 'Shear_Rate', 'Temperature', 'PDI', 'Melt_Viscosity', 'SMILES'])
    
    formatted_data = []
    
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_fp)
    for i, row in df_raw.iterrows():
        # Handle missing values where necessary, but we need targets
        if pd.isna(row['Melt_Viscosity']) or pd.isna(row['Temperature']):
            continue
            
        smiles_list = [s.strip().strip('"') for s in str(row['SMILES']).split(',')]
        ptype = str(row['Sample_Type']).lower()
        
        weights = []
        if pd.notna(row['Weight 1']):
            weights.append(float(row['Weight 1']))
        if pd.notna(row['Weight 2']):
            weights.append(float(row['Weight 2']))
            
        if len(weights) == 0:
            weights = [1.0] # Homopolymer
            
        components = []
        valid_row = True
        
        for idx, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                valid_row = False
                break
            
            fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
            w = weights[idx] if idx < len(weights) else 0.0
            
            components.append({
                "fp": fp,
                "w": w
            })
            
        if not valid_row:
            continue
            
        formatted_data.append({
            'monomer_id': str(row['Polymer']),
            'polymer_type': ptype,
            'components': components,
            'Mw': float(row['Mw']) if pd.notna(row['Mw']) else np.nan,
            'PDI': float(row['PDI']) if pd.notna(row['PDI']) else np.nan,
            'T': float(row['Temperature']),
            'shear_rate': float(row['Shear_Rate']) if pd.notna(row['Shear_Rate']) else 0.0,
            'viscosity': float(row['Melt_Viscosity'])
        })
        
    df = pd.DataFrame(formatted_data)
    print(f"Parsed {len(df)} valid records.")
    return df

@torch.no_grad()
def evaluate_penn(model, dataloader, criterion, device):
    """
    Evaluates the model on dataloader, computes regression metrics,
    and extracts physics constants distributions.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    collected_constants = {
        'alpha_1': [], 'alpha_2': [], 'M_cr': [], 'k_1': [],
        'C1': [], 'C2': [], 'T_r': [], 'S_cr': [],
        'n': [], 'Beta_Mw': [], 'Beta_Shear': []
    }
    
    for batch_idx, (XX, M, S, T, P, visc) in enumerate(dataloader):
        XX = XX.to(device)
        M = M.to(device)
        S = S.to(device)
        T = T.to(device)
        P = P.to(device)
        visc = visc.to(device)
        
        out = model(XX, M, S, T, P)
        loss = criterion(out, visc)
        total_loss += loss.item() * len(visc)
        
        preds_np = out.squeeze().detach().cpu().numpy().reshape(-1)
        targets_np = visc.squeeze().detach().cpu().numpy().reshape(-1)
        all_preds.extend(preds_np.tolist())
        all_targets.extend(targets_np.tolist())
        
        # Extract constants
        consts = model(XX, M, S, T, P, get_constants=True)
        for k, v in consts.items():
            if k in collected_constants and v is not None:
                val_np = v.squeeze().detach().cpu().numpy().reshape(-1)
                collected_constants[k].extend(val_np.tolist())
                
    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    r2 = float(r2_score(all_targets, all_preds)) if len(all_preds) > 1 else 0.0
    mae = float(mean_absolute_error(all_targets, all_preds))
    rmse = float(np.sqrt(mean_squared_error(all_targets, all_preds)))
    
    constants_metrics = {}
    for k, v in collected_constants.items():
        if len(v) > 0:
            arr = np.array(v)
            constants_metrics[f"constants/{k}_mean"] = float(np.mean(arr))
            constants_metrics[f"constants/{k}_std"] = float(np.std(arr))
            constants_metrics[f"constants/{k}_min"] = float(np.min(arr))
            constants_metrics[f"constants/{k}_max"] = float(np.max(arr))
            
    metrics = {
        "val_loss": avg_loss,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "predictions": all_preds,
        "targets": all_targets,
        "constants": constants_metrics
    }
    
    return avg_loss, metrics

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    dataset_path = 'data/raw/melt_viscosity_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
        
    # 1. Load and format to pipeline specification
    df = load_and_format_dataset(dataset_path, n_fp=1024)
    
    # 2. Preprocess (aggregate fps, impute PDI, augment, scale)
    print("Running preprocessing pipeline...")
    df_processed, scaler = run_preprocessing_pipeline(df, is_train=True)
    
    # 3. Splitting Strategy (Extrapolation Probe)
    print("Generating extrapolation splits...")
    splits = generate_extrapolation_splits(df_processed, n_repeats=1)
    
    # Take the first split for this run (Mw extrapolation split)
    split_0 = splits[0]
    print(f"Using split on variable: {split_0['split_var']} | Train size: {len(split_0['train_idx'])}, Test size: {len(split_0['test_idx'])}")
    
    train_df, val_df = get_train_test_dataframes(df_processed, split_0)
    
    # 4. DataLoaders
    batch_size = 32
    print("Creating DataLoaders...")
    train_loader, val_loader = get_dataloaders(train_df, val_df, batch_size=batch_size)
    
    config = {
        "l1": 120,
        "l2": 120,
        "d1": 0.2,
        "d2": 0.2,
        "a_weight": 0.1
    }
    
    # 5. Initialize Weights & Biases
    print("Initializing Weights & Biases...")
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "polymer-viscosity-penn"),
        name=f"penn_extrapolation_{split_0['split_var']}",
        tags=["PENN", "melt-viscosity", "physics-informed", split_0['split_var']],
        config={
            "architecture": "Visc_PENN",
            "n_fp": 1024,
            "l1": config["l1"],
            "l2": config["l2"],
            "d1": config["d1"],
            "d2": config["d2"],
            "a_weight": config["a_weight"],
            "learning_rate": 1e-3,
            "optimizer": "Adam",
            "loss_fn": "MSELoss",
            "batch_size": batch_size,
            "max_epochs": 200,
            "early_stopping_patience": 20,
            "early_stopping_delta": 1e-4,
            "split_variable": split_0['split_var'],
            "train_size": len(train_df),
            "val_size": len(val_df),
            "device": str(device)
        }
    )
    
    # 6. Initialize Model & Training Setup
    print("Initializing model...")
    model = Visc_PENN(n_fp=1024, config=config, device=device, run=run, fold=0).to(device)
    
    # Log model topology and gradients to W&B
    wandb.watch(model, log="all", log_freq=10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=30, delta=1e-4)
    
    num_epochs = 200
    best_val_loss = float('inf')
    best_epoch = 0
    best_metrics = {}
    best_model_weights = None
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss = model.train_model(train_loader, optimizer, criterion, config)
        avg_train_loss = float(train_loss.get('avg_train_loss', 0))
        tot_train_loss = float(train_loss.get('epoch_train_loss', 0))
        a1_loss = float(train_loss.get('a1_loss', 0))
        a2_loss = float(train_loss.get('a2_loss', 0))
        print(f"Train Loss: {avg_train_loss:.4f}")
        
        # Detailed Evaluation
        avg_val_loss, val_metrics = evaluate_penn(model, val_loader, criterion, device)
        print(f"Val Loss: {avg_val_loss:.4f} | R²: {val_metrics['r2']:.4f} | RMSE: {val_metrics['rmse']:.4f} | MAE: {val_metrics['mae']:.4f}")
        
        # Log metrics to W&B
        log_payload = {
            "epoch": epoch + 1,
            "train/avg_loss": avg_train_loss,
            "train/tot_loss": tot_train_loss,
            "train/a1_loss": a1_loss,
            "train/a2_loss": a2_loss,
            "val/loss": avg_val_loss,
            "val/r2": val_metrics["r2"],
            "val/rmse": val_metrics["rmse"],
            "val/mae": val_metrics["mae"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "early_stopping/counter": early_stopping.counter,
            "early_stopping/best_loss": early_stopping.best_loss if early_stopping.best_loss != float('inf') else avg_val_loss,
            **val_metrics["constants"]
        }
        wandb.log(log_payload)
        
        # Track best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_metrics = val_metrics
            best_model_weights = model.state_dict().copy()
            
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}! Best Val Loss: {early_stopping.best_loss:.4f}")
            break
            
    print("\nTraining complete!")
    
    # 7. Post-Training W&B Logging (Summary, Table, Scatter Plot, Model Artifact)
    print("Uploading comprehensive evaluation artifacts to Weights & Biases...")
    
    # Save & Log Summary
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary["best_val_r2"] = best_metrics.get("r2", 0.0)
    wandb.summary["best_val_rmse"] = best_metrics.get("rmse", 0.0)
    wandb.summary["best_val_mae"] = best_metrics.get("mae", 0.0)
    
    # Create Detailed Predictions Table
    best_preds = np.asarray(best_metrics.get("predictions", []))
    best_targets = np.asarray(best_metrics.get("targets", []))
    
    table_data = []
    val_reset = val_df.reset_index(drop=True)
    for idx in range(len(best_preds)):
        actual = float(best_targets[idx])
        pred = float(best_preds[idx])
        residual = actual - pred
        abs_err = abs(residual)
        
        row_dict = val_reset.iloc[idx] if idx < len(val_reset) else {}
        monomer = str(row_dict.get('monomer_id', f'Sample_{idx}'))
        ptype = str(row_dict.get('polymer_type', 'unknown'))
        mw = float(row_dict.get('Mw', 0.0))
        shear = float(row_dict.get('shear_rate', 0.0))
        temp = float(row_dict.get('T', 0.0))
        pdi = float(row_dict.get('PDI', 0.0))
        
        table_data.append([
            idx, monomer, ptype, mw, shear, temp, pdi, actual, pred, residual, abs_err
        ])
        
    val_table = wandb.Table(
        columns=["Index", "Monomer", "Polymer_Type", "Mw_scaled", "Shear_scaled", "T_scaled", "PDI_scaled", "Actual_Viscosity", "Predicted_Viscosity", "Residual", "Absolute_Error"],
        data=table_data
    )
    wandb.log({"val_predictions_table": val_table})
    
    # Log Scatter Plot of Actual vs Predicted
    wandb.log({
        "actual_vs_predicted": wandb.plot.scatter(
            val_table, "Actual_Viscosity", "Predicted_Viscosity", title="Actual vs Predicted Viscosity (Validation Set)"
        )
    })
    
    # Save Model Weights & Log as WandB Artifact
    checkpoint_path = "best_penn_model.pth"
    if best_model_weights is not None:
        torch.save(best_model_weights, checkpoint_path)
        model_artifact = wandb.Artifact(
            name="penn_best_model",
            type="model",
            description="Best weights checkpoint for Physics Enforced Neural Network",
            metadata={
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_r2": best_metrics.get("r2", 0.0),
                "best_val_rmse": best_metrics.get("rmse", 0.0),
                "best_val_mae": best_metrics.get("mae", 0.0),
            }
        )
        model_artifact.add_file(checkpoint_path)
        wandb.log_artifact(model_artifact)
        print(f"Saved best model checkpoint from epoch {best_epoch} to {checkpoint_path} and logged to W&B.")
        
    wandb.finish()
    print("Weights & Biases logging finished successfully.")

if __name__ == '__main__':
    main()
