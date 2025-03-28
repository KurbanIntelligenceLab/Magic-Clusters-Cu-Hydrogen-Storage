import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader_feautre_kg_only import create_data_generator
from model import MultimodalModel
import wandb

materials_to_test = ['Cu_R7_optimized', 'Cu_R8_optimized', 'Cu_R9_optimized', 'Cu_R10_optimized']
seed = 42
num_of_folds = 3
batch_size = 48
num_epochs = 5
main_experiment_dir = 'results/image_feature_kg'
project_name = 'hydrogen_storage_knowledge_graph_experiments'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "materials_to_test": materials_to_test,
    "seed": seed,
    "num_of_folds": num_of_folds,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "optimizer": "Adam",
    "optimizer_params": {
         "lr": 1e-3,
         "weight_decay": 0
    },
    "lr_scheduler": {
         "step_size": 10,
         "gamma": 0.1
    },
    "dropout_rate": 0.2,
    "project_name": project_name
}

for material in materials_to_test:
    print(f"\nProcessing material: {material}")
    
    fold_loaders, test_loader = create_data_generator(material, seed, num_of_folds=num_of_folds, batch_size=batch_size)
    
    for fold, (train_loader, val_loader) in enumerate(fold_loaders, start=1):
        print(f"\nMaterial: {material} | Fold: {fold}/{num_of_folds}")
        
        exp_dir = os.path.join(main_experiment_dir, material, str(fold))
        os.makedirs(exp_dir, exist_ok=True)
        
        run_config = config.copy()
        run_config.update({"material": material, "fold": fold})
        
        wandb.init(project=project_name, config=run_config, reinit=True)
        
        sample = train_loader.dataset[0]
        num_graph_features = sample['features'].shape[0]
        
        model = MultimodalModel(num_graph_features=num_graph_features, num_graph_outputs=4, num_outputs=1, dropout_rate=run_config["dropout_rate"]).to(device)
        criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), **run_config["optimizer_params"])
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=run_config["lr_scheduler"]["step_size"],
                                                 gamma=run_config["lr_scheduler"]["gamma"])
        
        best_val_loss = float('inf')
        best_model_path = os.path.join(exp_dir, "best_model.pt")
        train_losses = []
        val_losses = []
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            model.train()
            running_train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                images = batch['image'].to(device)
                graph_features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, graph_features)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                
                running_train_loss += loss.item() * images.size(0)
            
            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)
            
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    graph_features = batch['features'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(images, graph_features)
                    loss = criterion(outputs.squeeze(), labels)
                    running_val_loss += loss.item() * images.size(0)
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
            
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - LR: {current_lr:.6f}")
            
            wandb.log({
                "epoch": epoch+1,
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "lr": current_lr
            })
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), best_model_path)
                wandb.run.summary["best_val_loss"] = best_val_loss
                print("  Best model saved.")
        
        training_duration = time.time() - start_time
        
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                graph_features = batch['features'].to(device)
                labels = batch['label'].to(device)
                outputs = model(images, graph_features)
                loss = criterion(outputs.squeeze(), labels)
                running_test_loss += loss.item() * images.size(0)
        test_loss = running_test_loss / len(test_loader.dataset)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Training Duration (seconds): {training_duration:.2f}")
        
        # Log test loss and training duration, and add them to the summary
        wandb.log({"test_loss": test_loss, "training_duration": training_duration})
        wandb.run.summary["test_loss"] = test_loss
        wandb.run.summary["training_duration"] = training_duration
        
        results = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "test_loss": test_loss,
            "training_duration": training_duration
        }
        
        results_path = os.path.join(exp_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
        
        wandb.finish()
