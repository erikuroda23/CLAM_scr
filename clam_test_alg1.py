import torch
import matplotlib.pyplot as plt
import os
from stage1 import IDM, FDM, ActionDecoder, VideoActuatorDataset, transform

# path
model_dir = "step_10000_k5/trained_models"
task_name = "all_tasks" 
test_name = "walker_run_test"
video_path = f"datasets/{test_name}/test_0_video.mp4"
npz_path = f"datasets/{test_name}/test_0_actions.npz"

# model
input_dim = 3 * 64 * 64
latent_dim = 32
action_dim = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load models
idm = IDM(input_dim, latent_dim).to(device)
idm.load_state_dict(torch.load(os.path.join(model_dir, f"{task_name}_idm.pt")))
idm.eval()

fdm = FDM(input_dim, latent_dim).to(device)
fdm.load_state_dict(torch.load(os.path.join(model_dir, f"{task_name}_fdm.pt")))
fdm.eval()

decoder = ActionDecoder(latent_dim, action_dim).to(device)
decoder.load_state_dict(torch.load(os.path.join(model_dir, f"{task_name}_decoder.pt")))
decoder.eval()

# dataloader
dataset = VideoActuatorDataset(video_path, npz_path=npz_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# save
save_dir = f"step_10000_k5/test_result/{task_name}"
os.makedirs(save_dir, exist_ok=True)

# loss
mse = torch.nn.MSELoss()
total_recon_loss = 0
total_action_loss = 0

for i, (o_t, a_t, o_tp1) in enumerate(dataloader):
    o_t = o_t.to(device).view(1, -1)
    o_tp1 = o_tp1.to(device).view(1, -1)
    a_t = a_t.to(device)

    with torch.no_grad():
        z_t = idm(o_t, o_tp1)
        o_tp1_hat = fdm(o_t, z_t)
        a_hat = decoder(z_t)

        # MSE
        loss_obs = mse(o_tp1_hat, o_tp1)
        loss_action = mse(a_hat, a_t)

        total_recon_loss += loss_obs.item()
        total_action_loss += loss_action.item()

        # plot
        img_gt = o_tp1.view(3, 64, 64).cpu().permute(1, 2, 0).numpy()
        img_pred = o_tp1_hat.view(3, 64, 64).cpu().permute(1, 2, 0).numpy()

        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(img_gt)
        axs[0].set_title("Ground Truth")
        axs[1].imshow(img_pred)
        axs[1].set_title("Reconstruction")
        for ax in axs: ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"step_{i:04d}.png"))
        plt.close()

    if i >= 9:  # 最初の10ステップだけ確認
        break

# output loss
print(f"Mean Reconstruction Loss: {total_recon_loss / 10:.4f}")
print(f"Mean Action Loss: {total_action_loss / 10:.4f}")
