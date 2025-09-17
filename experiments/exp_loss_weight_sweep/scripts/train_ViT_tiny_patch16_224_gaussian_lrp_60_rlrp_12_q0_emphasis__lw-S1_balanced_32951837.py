#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import json

from config import get_config
from train_utils.gpu_utils import get_device_summary
from data.loader import get_dataloaders
from models.model import create_model
from train_utils.resume import init_resume_state
from train_utils.resume import fill_trackers_from_history
from train_utils.resume import load_pretrained_model
from train_utils.training_loop import run_training_loop
from train_utils.scheduler_utils import create_scheduler
from train_utils.training_summary import finalize_training_summary
from train_utils.training_summary import print_best_model_summary
from train_utils.plot_metrics import plot_train_val_metrics
from train_utils.plot_metrics import plot_loss_accuracy
from train_utils.plot_metrics import plot_confusion_matrices


# In[ ]:


# cfg=get_config(config_path="config/convnext_fb_in22k_ft_in1k_bs512_ep50_lr1e-04_ds1000.yml")
# cfg=get_config(config_path="config/convnext_fb_in1k_bs512_ep50_lr1e-04_ds1000.yml")
# cfg=get_config(config_path="config/convnext_gaussian_bs512_ep50_lr1e-04_ds1000.yml")
# cfg=get_config(config_path="config/efficientnet_bs512_ep50_lr1e-01_ds1000_sched-RLRP.yml")
# cfg=get_config(config_path="config/vit_" \
# "bs512_ep50_lr1e-04_ds1000.yml")
# cfg=get_config(config_path="config/mambaout_base_plus_rw_bs32_ep50_lr1e-04_ds1000-g1.yml")
# cfg=get_config(config_path="config/mambaout_base_plus_rw_bs16_ep50_lr1e-04_ds1008_g500_sched-RLRP.yml")

# from experiments.exp_mamaba_vit_stack.models.hybrid_mamba_vit import create_model
# cfg=get_config(config_path="/home/arsalan/wsu-grid/ml-jet-param-predictor/" \
# "experiments/exp_mamaba_vit_stack/config/" \
# "hybrid_mambaout_base_plus_rw_ViT_tiny_patch16_224_bs64_ep1_lr1e-04_ds1008_g500_sched-RLRP.yml")
# from models.model_mamba import create_model
from models.model_vit import create_model
# cfg=get_config(config_path="/home/arsalan/wsu-grid/ml-jet-param-predictor/" \
# "experiments/exp_adding_loss_weights_for_q0_emphasis/config/" \
# "vit_tiny_patch16_224_gaussian_bs32_ep200_lr1e-04_p60_ds1008_g500_sched-RLRP_preload_p12.yml")
# from models.model import create_model
# cfg=get_config(config_path="/home/arsalan/wsu-grid/ml-jet-param-predictor/" \
# "experiments/exp_loss_weight_sweep/configs/" \
# "convnext_gaussian_bs32_ep50_lr1e-04_ds7200000_g500_sched-RLRP__lw-S1_balanced.yml")

cfg=get_config()
print(json.dumps(vars(cfg), indent=2))


# In[ ]:


os.makedirs(cfg.output_dir, exist_ok=True)
print(f"[INFO] Saving all outputs to: {cfg.output_dir}")


# In[ ]:


device= get_device_summary()


# In[ ]:


# Data
train_loader, val_loader, test_loader = get_dataloaders(cfg, device=device)


# In[ ]:


# Model and optimizer
model, optimizer = create_model(cfg.backbone, cfg.input_shape, cfg.learning_rate)
model.to(device)


# In[ ]:


if torch.cuda.device_count() > 1:
    print(f"Parallelizing model across {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
elif torch.cuda.device_count() == 1:
    print("No parallelization, using single GPU")
elif torch.cuda.device_count() == 0:
    print("No GPU available, using CPU")


# In[ ]:


scheduler = create_scheduler(optimizer, cfg, train_loader=train_loader)


# In[ ]:


criterion = {
    # 'energy_loss_output': nn.BCELoss(),
    'energy_loss_output': nn.BCEWithLogitsLoss(),
    'alpha_output': nn.CrossEntropyLoss(),
    'q0_output': nn.CrossEntropyLoss()
}
print(f"[INFO] Loss functions:{criterion}")


# In[ ]:


print(f"[INFO] Init Training Trackers")
train_loss_energy_list, train_loss_alpha_list, train_loss_q0_list, train_loss_list = [], [], [],[]
train_acc_energy_list, train_acc_alpha_list, train_acc_q0_list, train_acc_list = [], [], [], []

print(f"[INFO] Init Validation Trackers")
val_loss_energy_list, val_loss_alpha_list,val_loss_q0_list,val_loss_list = [], [], [], []
val_acc_energy_list, val_acc_alpha_list,val_acc_q0_list ,val_acc_list = [],[],[],[]


# In[ ]:


model, optimizer, start_epoch, best_acc, early_stop_counter, best_epoch, best_metrics, training_summary, all_epoch_metrics,summary_status = init_resume_state( model, optimizer, device,cfg)


# In[ ]:


fill_trackers_from_history(
    all_epoch_metrics,
    train_loss_energy_list, train_loss_alpha_list, train_loss_q0_list, train_loss_list,
    train_acc_energy_list, train_acc_alpha_list, train_acc_q0_list, train_acc_list,
    val_loss_energy_list, val_loss_alpha_list, val_loss_q0_list, val_loss_list,
    val_acc_energy_list, val_acc_alpha_list, val_acc_q0_list, val_acc_list,
    summary_status, best_epoch
)


# In[ ]:


model, preloaded = load_pretrained_model(model, device, cfg)


# In[ ]:


# for testing
# train_metrics = train_one_epoch(train_loader, model, criterion, optimizer, device)
# print(f"[INFO] Training metrics: {train_metrics}")


# In[ ]:


best_epoch,best_acc,best_metrics=run_training_loop(
                      cfg,train_loader,val_loader,
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
                        all_epoch_metrics)


# In[ ]:


finalize_training_summary(
    summary=training_summary,
    best_epoch=best_epoch,
    best_acc=best_acc,
    best_metrics=best_metrics,
    output_dir=cfg.output_dir
)
print_best_model_summary(
    best_epoch=best_epoch,
    best_acc=best_acc,
    best_metrics=best_metrics
)


# In[ ]:


plot_train_val_metrics(train_loss_list, val_loss_list, train_acc_list, val_acc_list, cfg.output_dir)
plot_loss_accuracy(train_loss_list,
                    train_loss_energy_list,
                    train_loss_alpha_list,
                    train_loss_q0_list,
                    train_acc_list,
                    train_acc_energy_list,
                    train_acc_alpha_list,
                    train_acc_q0_list,
                    cfg.output_dir,
                    title="Train Loss and Accuracy per Epoch")
plot_loss_accuracy(val_loss_list,
                    val_loss_energy_list,
                    val_loss_alpha_list,
                    val_loss_q0_list,
                    val_acc_list,
                    val_acc_energy_list,
                    val_acc_alpha_list,
                    val_acc_q0_list,
                    cfg.output_dir,
                    title="Validation Loss and Accuracy per Epoch")


# In[ ]:


plot_confusion_matrices(best_metrics, output_dir= f"{cfg.output_dir}/val", color_map="Oranges")


# In[ ]:


alpha_names = ("0.2","0.3","0.4")
q0_names    = ("1.0","1.5","2.0","2.5")


# In[ ]:


from train_utils.evaluate import evaluate
# experiment folder for artifacts
art_dir = os.path.join(cfg.output_dir,"val/prob_plots")
os.makedirs(art_dir, exist_ok=True)

alpha_hist_path      = os.path.join(art_dir, "alpha_pred_hist")      # .png/.pdf added by evaluate
alpha_avgprob_path   = os.path.join(art_dir, "alpha_avgprob")        # .png/.pdf added by evaluate
q0_avgprob_path      = os.path.join(art_dir, "q0_avgprob")           # .png/.pdf added by evaluate

metrics_val = evaluate(
    val_loader, model, criterion, device,
    loss_weights=getattr(cfg, "loss_weights", None),
    # plots:
    make_alpha_fig=True,
    alpha_fig_path=str(alpha_hist_path),
    make_alpha_avgprob_fig=True,
    alpha_avgprob_fig_path=str(alpha_avgprob_path),
    make_q0_avgprob_fig=True,
    q0_avgprob_fig_path=str(q0_avgprob_path),
    alpha_class_names=alpha_names,
    q0_class_names=q0_names,
)

print("Saved images:")
print("  α_s histogram:        ", metrics_val.get("alpha_hist_path"))
print("  α_s avg-prob bars:    ", metrics_val.get("alpha_avgprob_hist_path"))
print("  α_s probabilities:     ", metrics_val.get("alpha_probs_csv"))
print("  α_s heatmap:          ", metrics_val.get("alpha_heatmap_path"))
print("  Q0  avg-prob bars:    ", metrics_val.get("q0_avgprob_hist_path"))
print("  Q0  probabilities:     ", metrics_val.get("q0_probs_csv"))
print("  Q0  heatmap:          ", metrics_val.get("q0_heatmap_path"))


# In[ ]:


# (optional) test split as well
try:
    from train_utils.evaluate import evaluate
    # experiment folder for artifacts
    art_dir = os.path.join(cfg.output_dir,"test/prob_plots")
    os.makedirs(art_dir, exist_ok=True)

    alpha_hist_path      = os.path.join(art_dir, "alpha_pred_hist")      # .png/.pdf added by evaluate
    alpha_avgprob_path   = os.path.join(art_dir, "alpha_avgprob")        # .png/.pdf added by evaluate
    q0_avgprob_path      = os.path.join(art_dir, "q0_avgprob")           # .png/.pdf added by evaluate

    metrics_test = evaluate(
        test_loader, model, criterion, device,
        loss_weights=getattr(cfg, "loss_weights", None),
        # plots:
        make_alpha_fig=True,
        alpha_fig_path=str(alpha_hist_path),
        make_alpha_avgprob_fig=True,
        alpha_avgprob_fig_path=str(alpha_avgprob_path),
        make_q0_avgprob_fig=True,
        q0_avgprob_fig_path=str(q0_avgprob_path),
        alpha_class_names=alpha_names,
        q0_class_names=q0_names,
    )

    print("Saved images:")
    print("  α_s histogram:        ", metrics_test.get("alpha_hist_path"))
    print("  α_s avg-prob bars:    ", metrics_test.get("alpha_avgprob_hist_path"))
    print("  α_s probabilities:     ", metrics_test.get("alpha_probs_csv"))
    print("  α_s heatmap:          ", metrics_test.get("alpha_heatmap_path"))
    print("  Q0  avg-prob bars:    ", metrics_test.get("q0_avgprob_hist_path"))
    print("  Q0  probabilities:     ", metrics_test.get("q0_probs_csv"))
    print("  Q0  heatmap:          ", metrics_test.get("q0_heatmap_path"))
    have_test = True
except NameError:
    have_test = False

