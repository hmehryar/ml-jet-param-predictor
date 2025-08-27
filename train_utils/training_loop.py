from train_utils.train_epoch import train_one_epoch
from train_utils.train_metrics_logger import update_train_logs
from train_utils.evaluate import evaluate
from train_utils.train_metrics_logger import update_val_logs
from train_utils.train_metrics_logger import record_and_save_epoch
from train_utils.early_stopping import check_early_stopping

def run_training_loop(cfg,train_loader,val_loader,
                      device, model,criterion,
                      optimizer,scheduler,
                      start_epoch,early_stop_counter,
                      best_acc,best_metrics,best_epoch,
                      train_loss_list,
                        train_loss_energy_list,
                        train_loss_alpha_list,
                        train_loss_q0_list,
                        train_acc_list,
                        train_acc_energy_list,
                        train_acc_alpha_list,
                        train_acc_q0_list,
                        val_loss_list,
                        val_loss_energy_list,
                        val_loss_alpha_list,
                        val_loss_q0_list,
                        val_acc_list,
                        val_acc_energy_list,
                        val_acc_alpha_list,
                        val_acc_q0_list,
                        all_epoch_metrics):
    for epoch in range(start_epoch, cfg.epochs):
        print(f"[INFO] Epoch {epoch+1}/{cfg.epochs}")
        train_metrics={}
        train_metrics = train_one_epoch(train_loader, model, criterion, optimizer, device,
                                        loss_weights=cfg.loss_weights)
        (train_loss_list,
        train_loss_energy_list,
        train_loss_alpha_list,
        train_loss_q0_list,
        train_acc_list,
        train_acc_energy_list,
        train_acc_alpha_list,
        train_acc_q0_list
        ) = update_train_logs(
            train_metrics,
            train_loss_list,
            train_loss_energy_list,
            train_loss_alpha_list,
            train_loss_q0_list,
            train_acc_list,
            train_acc_energy_list,
            train_acc_alpha_list,
            train_acc_q0_list
        )
        # ensure evaluate() computes weighted total loss with same weights
        val_metrics = evaluate(val_loader, model, criterion, device, loss_weights=cfg.loss_weights)
        (val_loss_list,
        val_loss_energy_list,
        val_loss_alpha_list,
        val_loss_q0_list,
        val_acc_list,
        val_acc_energy_list,
        val_acc_alpha_list,
        val_acc_q0_list,
        ) = update_val_logs(
            val_metrics,
            val_loss_list,
            val_loss_energy_list,
            val_loss_alpha_list,
            val_loss_q0_list,
            val_acc_list,
            val_acc_energy_list,
            val_acc_alpha_list,
            val_acc_q0_list,
        )
        print(f"[INFO] Epoch {epoch+1}: Energy Acc ={val_metrics['energy']['accuracy']:.4f}, αs Acc = {val_metrics['alpha']['accuracy']:.4f}, Q0 Acc = {val_metrics['q0']['accuracy']:.4f}, Total Acc = {val_metrics['accuracy']:.4f}")
        print(f"[INFO] Epoch {epoch+1}: Energy Loss ={val_metrics['loss_energy']:.4f}, αs Loss = {val_metrics['loss_alpha']:.4f}, Q0 Loss = {val_metrics['loss_q0']:.4f}, Total Loss = {val_metrics['loss']:.4f}")
        
        scheduler.step(val_metrics['accuracy'])  # or macro average accuracy if defined
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            print(f"📉 Current LR: {current_lr}")
        
        all_epoch_metrics=record_and_save_epoch(epoch, train_metrics, val_metrics, current_lr, all_epoch_metrics, cfg.output_dir)
        
        # save_epoch_checkpoint(
        #     epoch=epoch,
        #     model=model,
        #     optimizer=optimizer,
        #     metrics=val_metrics,
        #     output_dir=cfg.output_dir
        # )

        best_acc, best_metrics, best_epoch, early_stop_counter, should_stop = check_early_stopping(
            best_acc=best_acc,
            best_metrics=best_metrics,
            early_stop_counter=early_stop_counter,
            best_epoch=best_epoch,
            model=model,
            optimizer=optimizer,
            val_metrics=val_metrics,
            output_dir=cfg.output_dir,
            patience=cfg.patience,
            epoch=epoch
        )
        
        if should_stop:
            break
        
        print("="*150)
    return best_epoch,best_acc,best_metrics
        