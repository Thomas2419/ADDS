import time
import torch
from torch.optim.lr_scheduler import LambdaLR
import os
from torch.utils.tensorboard import SummaryWriter
import open_clip
from model import ADDSModel

def load_model(model_path, device, clip_model, embed_dim, projected_dim, num_heads, num_layers, num_labels):
    model = ADDSModel(clip_model=clip_model, embed_dim=embed_dim, projected_dim=projected_dim, num_heads=num_heads, num_layers=num_layers, num_labels=num_labels).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    return model

def save_checkpoint(model, optimizer, epoch, epoch_acc, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_acc_{epoch_acc:.6f}.pth")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")

def calculate_accuracy(outputs, labels):
    probas = torch.sigmoid(outputs)
    thresholds = torch.tensor([0.6, 0.7, 0.8, 0.9], device=outputs.device).view(-1, 1)
    preds = probas.unsqueeze(0) > thresholds.unsqueeze(1)
    corrects = (preds == labels.unsqueeze(0)).float()
    accuracies = corrects.mean(dim=(1, 2)).tolist()
    return dict(zip([0.6, 0.7, 0.8, 0.9], accuracies))

def linear_warmup_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0  # Constant learning rate after warmup

    return LambdaLR(optimizer, lr_lambda)

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=25, checkpoint_dir="checkpoints", text_embeddings=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = None
    best_accuracies = {0.6: 0.0, 0.7: 0.0, 0.8: 0.0, 0.9: 0.0}
    scaler = torch.cuda.amp.GradScaler()
    main_log_dir = "main_logs"
    os.makedirs(main_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(main_log_dir, time.strftime("%Y%m%d-%H%M%S")))
    total_steps = len(train_loader) * num_epochs
    scheduler = linear_warmup_lr_scheduler(optimizer, warmup_steps=100, total_steps=total_steps)
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = {0.6: 0.0, 0.7: 0.0, 0.8: 0.0, 0.9: 0.0}
        total_samples = 0
        epoch_start_time = time.time()

        # Training Loop
        for i, (images, labels) in enumerate(train_loader):  # Removed patch_indices
            step_start_time = time.time()
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            torch.compile(model)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(images, text_embeddings=text_embeddings)  # Removed patch_indices
                
                # Print shapes of outputs and labels during validation
                #print(f"Training: Outputs shape: {outputs.shape}")
                #print(f"Labels shape: {labels.shape}")
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            accuracies = calculate_accuracy(outputs, labels)
            for threshold in running_corrects:
                running_corrects[threshold] += accuracies[threshold] * labels.size(0)

            total_samples += labels.size(0)
            global_step += 1
            step_time = time.time() - step_start_time
            steps_per_second = 1.0 / step_time

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], global_step)
            for threshold in running_corrects:
                writer.add_scalar(f'Accuracy/train@{threshold}', accuracies[threshold] * 100, global_step)

            progress = (f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                        f"Loss: {loss.item():.6}, Acc@0.6: {accuracies[0.6] * 100:.6}%, "
                        f"Acc@0.7: {accuracies[0.7] * 100:.6}%, Acc@0.8: {accuracies[0.8] * 100:.6}%, "
                        f"Acc@0.9: {accuracies[0.9] * 100:.6}%, Steps/s: {steps_per_second:.4}")
            print(progress, end='\r')

        epoch_time = time.time() - epoch_start_time
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracies = {threshold: running_corrects[threshold] / total_samples for threshold in running_corrects}

        print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6}, "
              f"Acc@0.6: {epoch_accuracies[0.6] * 100:.6}%, "
              f"Acc@0.7: {epoch_accuracies[0.7] * 100:.6}%, "
              f"Acc@0.8: {epoch_accuracies[0.8] * 100:.6}%, "
              f"Acc@0.9: {epoch_accuracies[0.9] * 100:.6}%, "
              f"Time: {epoch_time:.4}s")

        writer.add_scalar('Loss/epoch', epoch_loss, epoch)
        for threshold in epoch_accuracies:
            writer.add_scalar(f'Accuracy/epoch@{threshold}', epoch_accuracies[threshold] * 100, epoch)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_corrects = {0.6: 0.0, 0.7: 0.0, 0.8: 0.0, 0.9: 0.0}
        total_val_samples = 0
        with torch.no_grad():
            for images, labels in val_loader:  # Removed patch_indices
                images = images.to(device)
                labels = labels.to(device)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(images, text_embeddings=text_embeddings)  # Removed patch_indices
                    
                    # Print shapes of outputs and labels during validation
                    #print(f"Validation: Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
                    
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                accuracies = calculate_accuracy(outputs, labels)
                for threshold in val_corrects:
                    val_corrects[threshold] += accuracies[threshold] * labels.size(0)

                total_val_samples += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracies = {threshold: val_corrects[threshold] / total_val_samples for threshold in val_corrects}

        print(f"Validation Loss: {val_loss:.6}, "
              f"Val Acc@0.6: {val_accuracies[0.6] * 100:.6}%, "
              f"Val Acc@0.7: {val_accuracies[0.7] * 100:.6}%, "
              f"Val Acc@0.8: {val_accuracies[0.8] * 100:.6}%, "
              f"Val Acc@0.9: {val_accuracies[0.9] * 100:.6}%")

        writer.add_scalar('Loss/val', val_loss, epoch)
        for threshold in val_accuracies:
            writer.add_scalar(f'Accuracy/val@{threshold}', val_accuracies[threshold] * 100, epoch)

        # Save checkpoints
        if (epoch + 1) % 100 == 0:
            save_checkpoint(model, optimizer, epoch, val_accuracies[0.8], checkpoint_dir)

        for threshold in [0.6, 0.7, 0.8, 0.9]:
            if val_accuracies[threshold] > best_accuracies.get(threshold, 0):
                best_accuracies[threshold] = val_accuracies[threshold]
                best_checkpoint_path = os.path.join(checkpoint_dir, f"best_checkpoint_{threshold:.1f}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracies': best_accuracies,
                }, best_checkpoint_path)
                print(f"New best model saved with validation accuracy: {val_accuracies[threshold] * 100:.6}% at threshold {threshold} and epoch {epoch+1}")
                break
            elif val_accuracies[threshold] == best_accuracies.get(threshold, 0):
                continue
            else:
                break

    final_model_path = os.path.join(checkpoint_dir, "final_fine_tuned_open_clip_adds.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    writer.close()

def print_model_size(model):
    base_model_params = sum(p.numel() for p in model.base_model.parameters() if p.requires_grad)
    custom_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and "base_model" not in name)
    
    total_params = base_model_params + custom_params
    print(f"The total model size is approximately {total_params / 1e6:.4f} million parameters.")
    print(f"The base model (OpenCLIP) size is approximately {base_model_params / 1e6:.4f} million parameters.")
    print(f"The custom model size (excluding OpenCLIP) is approximately {custom_params / 1e6:.4f} million parameters.")


