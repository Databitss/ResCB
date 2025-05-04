import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             roc_curve, auc)
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from ResCB.model import *
from ResCB.data import *

class Res4NetCBAM():
    def __init__(self):
        self.model = Res4Net_CBAM(num_classes=3)

    def plot_training_metrics(self, all_results):
        """
        Plot training and validation losses and accuracies for all optimizer and learning rate combinations,
        with simplified legend (1 color per key).
        """
        epochs = range(1, self.epochs + 1)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
        random.shuffle(colors)

        simplified_handles = []
        simplified_labels = []

        for idx, (key, results) in enumerate(all_results.items()):
            color = colors[idx % len(colors)]
            train_losses = results["train_losses"]
            val_losses = results["val_losses"]
            train_accuracies = results["train_accuracies"]
            val_accuracies = results["val_accuracies"]

            # Plot all curves with same color
            axes[0, 0].plot(epochs, train_losses, marker=".", color=color)
            axes[0, 1].plot(epochs, val_losses, marker=".", color=color)
            axes[1, 0].plot(epochs, train_accuracies, marker=".", color=color)
            axes[1, 1].plot(epochs, val_accuracies, marker=".", color=color)

            # Simpan satu handle untuk legenda
            handle, = axes[0, 0].plot([], [], color=color, label=key)
            simplified_handles.append(handle)
            simplified_labels.append(f"{key}")

        # Label & grid untuk tiap plot
        axes[0, 0].set_title("Training Loss")
        axes[0, 1].set_title("Validation Loss")
        axes[1, 0].set_title("Training Accuracy")
        axes[1, 1].set_title("Validation Accuracy")
        
        for ax in axes.flat:
            ax.set_xlabel("Epochs")
            ax.grid(True)

        axes[0, 0].set_ylabel("Loss")
        axes[1, 0].set_ylabel("Accuracy (%)")

        # Letakkan legenda di bawah
        fig.legend(simplified_handles, simplified_labels, loc='upper center', bbox_to_anchor=(0.5, -0.03), ncol=4, title="Warna masing-masing kombinasi")

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()

    def plot_roc(self, all_results):
        n_classes = self.n_class  # Sesuaikan dengan jumlah kelas

        for key in all_results.keys():
            print(f"\nEvaluating ROC Curve for model: {key}")
            
            # Load model terbaik untuk konfigurasi ini
            model_path = f"{self.check_dir}/Res4Net_CBAM_{key}.pth"
            model = Res4Net_CBAM(num_classes=n_classes).to(self.device)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            all_labels = []
            all_probs = []

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    
                    # Simpan label asli
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Konversi output ke probabilitas dengan softmax
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())

            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)

            fpr = {}
            tpr = {}
            roc_auc = {}

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot ROC Curve
            plt.figure(figsize=(8, 6))
            
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], color=sns.color_palette("Set1")[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

            # Plot diagonal (no skill)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.title(f'ROC Curve for {key}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.show()

            # Print AUC per kelas
            for i in range(n_classes):
                print(f'Class {i} AUC: {roc_auc[i]:.2f}')

    def train(self, path_data, criterion, device, num_classes, num_epochs=100, batch_size=32, checkpoint_dir='checkpoints'):
        self.epochs = num_epochs
        self.n_class = num_classes
        self.check_dir = checkpoint_dir
        self.train_loader, self.val_loader = loader(path_data, batch_size)
        self.device = device
        optimizers = {
            'Adam': optim.Adam, # Dengan Adam 
            # Nanti ditambahkan AdaGrad Optimizer
            'SGD': lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9) # Dengan SGD (+ Momentum)
        }
        learning_rates = [0.001, 0.0001, 0.00001] # Learning rate yang diatur

        # State awal dari loss, acc, best_model dan semua result
        best_overall_loss = float('inf')
        best_overall_acc = 0.0
        best_model_path = None
        all_results = {}

        # Fungsi pengecekan dari direktori apakah sudah ada dan jika belum maka akan dibuat sebuah folder baru
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        for opt_name, opt_func in optimizers.items():
            for lr in learning_rates:
                print(f"Training with {opt_name} and LR={lr}")
                
                model = Res4Net_CBAM(num_classes=num_classes).to(device)
                optimizer = opt_func(model.parameters(), lr=lr)
                train_losses = []
                val_losses = []
                train_accuracies = []
                val_accuracies = []
                best_val_loss = float('inf')
                best_val_accuracy = 0.0
                
                for epoch in range(num_epochs):
                    model.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    
                    for inputs, labels in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        
                        running_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    
                    train_loss = running_loss / len(self.train_loader)
                    train_accuracy = correct / total * 100
                    train_losses.append(train_loss)
                    train_accuracies.append(train_accuracy)

                    model.eval()
                    val_running_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for inputs, labels in self.val_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            
                            val_running_loss += loss.item()
                            _, predicted = torch.max(outputs, 1)
                            val_total += labels.size(0)
                            val_correct += (predicted == labels).sum().item()
                    
                    val_loss = val_running_loss / len(self.val_loader)
                    val_accuracy = val_correct / val_total * 100
                    val_losses.append(val_loss)
                    val_accuracies.append(val_accuracy)

                    checkpoint_path = f"{checkpoint_dir}/Res4Net_CBAM_{opt_name}_lr{lr}.pth"
                    if val_loss < best_val_loss or val_accuracy > best_val_accuracy:
                        best_val_loss = val_loss
                        best_val_accuracy = val_accuracy
                        torch.save(model.state_dict(), checkpoint_path)
                        print(f"Model improved! Saving checkpoint at epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
                        
                        if val_loss < best_overall_loss or val_accuracy > best_overall_acc:
                            best_overall_loss = val_loss
                            best_overall_acc = val_accuracy
                            best_model_path = checkpoint_path
                            
                        self.model = model

                    print(f"Epoch [{epoch+1}/{num_epochs}]")
                    print(f"  Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
                    print(f"  Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_accuracy:.2f}%")
                
                torch.save(model.state_dict(), checkpoint_path)  
                print(f"Final model saved at {checkpoint_path}")
                
                all_results[f"{opt_name}_lr{lr}"] = {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_accuracies": train_accuracies,
                    "val_accuracies": val_accuracies
                }
        
        print(f"Best model saved at {best_model_path} with Val Loss: {best_overall_loss:.4f} | Val Accuracy: {best_overall_acc:.2f}%")
        return best_model_path, all_results

    def evaluation(self, all_results):
        def calculate_metrics(y_true, y_pred, num_classes):
            cm = confusion_matrix(y_true, y_pred)
            TN = np.diag(cm).sum() - np.diag(cm)   # True Negative
            FP = cm.sum(axis=0) - np.diag(cm)      # False Positive
            FN = cm.sum(axis=1) - np.diag(cm)      # False Negative
            TP = np.diag(cm)                       # True Positive
            
            accuracy = accuracy_score(y_true, y_pred) # Skor akurasi
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0) # Skor presisi
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)  # Skor recall (sensitivitas)
            specificity = np.mean(TN / (TN + FP + 1e-7))  # Skor spesifisitas
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) # F1-Score
            return accuracy, precision, recall, specificity, f1

        for key in all_results.keys():
            print(f"\nEvaluating model: {key}")
            
            # Load model terbaik untuk konfigurasi ini
            model_path = f"{self.check_dir}/Res4Net_CBAM_{key}.pth"
            model = Res4Net_CBAM(num_classes=3).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            all_labels = []
            all_preds = []

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())

            # Compute metrics
            accuracy, precision, recall, specificity, f1 = calculate_metrics(all_labels, all_preds, num_classes=3)
            
            # Print metrics
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Sensitivity (Recall): {recall:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print(f"F1-Score: {f1:.4f}")

            # Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(3), yticklabels=range(3))
            plt.title(f'Confusion Matrix for {key}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

        # Classification Report
        print(f"Classification Report for {key}")
        print(classification_report(all_labels, all_preds))
