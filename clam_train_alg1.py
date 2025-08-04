import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import cv2
from torchvision import transforms
import glob
import os
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Dataset
class VideoActuatorDataset(Dataset):
    def __init__(self, video_path: str, npz_path: str = None, transform=None):
        self.cap = cv2.VideoCapture(video_path)
        self.frames = []
        self.transform = transform
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            self.frames.append(frame)
        self.cap.release()

        self.labeled = npz_path is not None
        if self.labeled:
            self.actions = np.load(npz_path)["actions"]
            expected_frame_len = len(self.actions) + 1
            if len(self.frames) > expected_frame_len:
                self.frames = self.frames[:expected_frame_len]
            elif len(self.frames) < expected_frame_len:
                self.actions = self.actions[:len(self.frames) - 1]

    def __len__(self):
        return len(self.frames) - 1 if len(self.frames) > 1 else 0

    def __getitem__(self, idx):
        o_t = self.frames[idx]
        o_tp1 = self.frames[idx + 1]
        if self.labeled:
            a_t = self.actions[idx]
            return o_t, torch.tensor(a_t, dtype=torch.float32), o_tp1
        else:
            return o_t, o_tp1

# Models
class IDM(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, o_t, o_tp1):
        x = torch.cat([o_t, o_tp1], dim=1)
        return self.net(x)

class FDM(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, o_t, z_t):
        x = torch.cat([o_t, z_t], dim=1)
        return self.net(x)

class ActionDecoder(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, z_t):
        return self.net(z_t)

# Stage 1 training
def train_stage1(unlabeled_loader, labeled_loader, input_dim, action_dim, latent_dim=32, NC=10000, K=10, task_name="task"):
    idm = IDM(input_dim, latent_dim).to(device)
    fdm = FDM(input_dim, latent_dim).to(device)
    decoder = ActionDecoder(latent_dim, action_dim).to(device)

    optimizer = optim.Adam(list(idm.parameters()) + list(fdm.parameters()) + list(decoder.parameters()), lr=1e-4)
    mse = nn.MSELoss()
    labeled_iter = iter(labeled_loader)

    losses = []  #初期化

    for step in tqdm(range(1, NC + 1)):
        loss_action = torch.tensor(0.0).to(device)  #初期化

        for o_t, o_tp1 in unlabeled_loader:
            o_t = o_t.to(device).view(o_t.size(0), -1)
            o_tp1 = o_tp1.to(device).view(o_tp1.size(0), -1)

            z_t = idm(o_t, o_tp1)
            o_tp1_hat = fdm(o_t, z_t)
            loss_obs = mse(o_tp1_hat, o_tp1)
            loss = loss_obs

            # Stage 2 
            if step % K == 0:
                try:
                    o_t_l, a_t, o_tp1_l = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    o_t_l, a_t, o_tp1_l = next(labeled_iter)

                o_t_l = o_t_l.to(device).view(o_t_l.size(0), -1)
                o_tp1_l = o_tp1_l.to(device).view(o_tp1_l.size(0), -1)
                a_t = a_t.to(device)

                z_t_l = idm(o_t_l, o_tp1_l)
                a_hat = decoder(z_t_l)
                loss_action = mse(a_hat, a_t)
                loss += loss_action

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())  #loss

            break

        if step % 100 == 0:
            print(f"[{step}/{NC}] Obs Loss: {loss_obs.item():.4f}")
            print(f"[{step}/{NC}] Action Loss: {loss_action.item():.4f}")
            print(f"[{step}/{NC}] Total Loss: {loss.item():.4f}")

            # image
            image_dir = os.path.join("step_100000_k5/recon_images", task_name)
            os.makedirs(image_dir, exist_ok=True)

            # 可視化
            with torch.no_grad():
                img_true = o_tp1[0].detach().cpu().view(3, 64, 64)
                img_pred = o_tp1_hat[0].detach().cpu().view(3, 64, 64)

                img_true_np = img_true.permute(1, 2, 0).numpy()
                img_pred_np = img_pred.permute(1, 2, 0).numpy()

                fig, axs = plt.subplots(1, 2, figsize=(6, 3))
                axs[0].imshow(img_true_np)
                axs[0].set_title("Ground Truth")
                axs[1].imshow(img_pred_np)
                axs[1].set_title("Prediction")
                for ax in axs: ax.axis("off")
                plt.tight_layout()

                image_path = os.path.join(image_dir, f"recon_step{step}.png")
                plt.savefig(image_path)
                plt.close()

        # loss image
        plt.figure()
        plt.plot(losses, label='Total Loss', linestyle='-', marker='')
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve ({task_name})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        curve_dir = os.path.join("step_100000_k5/recon_images", task_name)
        os.makedirs(curve_dir, exist_ok=True)
        curve_path = os.path.join("step_100000_k5/recon_images", task_name, "loss_curve.png")
        plt.savefig(curve_path)
        plt.close()

    print("Stage 1 done")
    return idm, fdm, decoder

# Load all datasets
def load_datasets_from_folder(root_folder, transform=None):
    # humanoidはaction次元数違うので一旦除外
    # task_dirs = ["cheetah_run", "humanoid_run", "humanoid_walk", "walker_run", "walker_walk"]
    task_dirs = ["cheetah_run", "walker_run", "walker_walk"]
    datasets = defaultdict(dict)

    for task in task_dirs:
        video_files = sorted(glob.glob(os.path.join(root_folder, task, "train_*_video.mp4")))
        action_files = sorted(glob.glob(os.path.join(root_folder, task, "train_*_actions.npz")))

        unlabeled_list = []
        labeled_list = []

        for vfile, afile in zip(video_files, action_files):
            try:
                unlabeled_ds = VideoActuatorDataset(vfile, None, transform=transform)
                labeled_ds = VideoActuatorDataset(vfile, afile, transform=transform)

                if len(unlabeled_ds) > 0:
                    unlabeled_list.append(unlabeled_ds)
                if len(labeled_ds) > 0:
                    labeled_list.append(labeled_ds)

            except Exception as e:
                print(f"Skipping {vfile} due to error: {e}")
                continue

        if unlabeled_list and labeled_list:
            unlabeled_concat = ConcatDataset(unlabeled_list)
            labeled_concat = ConcatDataset(labeled_list)

            datasets[task]["unlabeled"] = DataLoader(unlabeled_concat, batch_size=32, shuffle=True)
            datasets[task]["labeled"] = DataLoader(labeled_concat, batch_size=32, shuffle=True)

            print(f"Loaded task '{task}': {len(unlabeled_list)} unlabeled, {len(labeled_list)} labeled datasets")

    return datasets

# Train all tasks with logging 
def train_all_tasks_with_logging(
    datasets,
    input_dim,
    action_dim,
    latent_dim=32,
    NC=10000,
    K=10,
    save_dir="step_100000_k5/trained_models",
    log_file="training_log.txt"
):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, log_file)

    with open(log_path, "w") as log:
        log.write(f"=== CLAM Stage 1 Training Log ({datetime.datetime.now()}) ===\n")

        for task_name, loaders in datasets.items():
            log.write(f"\n--- Training Task: {task_name} ---\n")
            print(f"\n=== Training task: {task_name} ===")

            unlabeled_loader = loaders["unlabeled"]
            labeled_loader = loaders["labeled"]

            try:
                idm, fdm, decoder = train_stage1(
                    unlabeled_loader=unlabeled_loader,
                    labeled_loader=labeled_loader,
                    input_dim=input_dim,
                    action_dim=action_dim,
                    latent_dim=latent_dim,
                    NC=NC,
                    K=K,
                    task_name=task_name
                )

                torch.save(idm.state_dict(), os.path.join(save_dir, f"{task_name}_idm.pt"))
                torch.save(fdm.state_dict(), os.path.join(save_dir, f"{task_name}_fdm.pt"))
                torch.save(decoder.state_dict(), os.path.join(save_dir, f"{task_name}_decoder.pt"))

                #latent action 可視化
                visualize_latent_actions(
                    idm=idm,
                    dataloader=unlabeled_loader,
                    input_dim=input_dim,
                    task_name=task_name,
                    save_path=f"step_100000_k5/latent_z/{task_name}_latent_pca.png"
                )

                log.write(f"Finished training {task_name}\n")
                print(f"Finished training task: {task_name}")

            except Exception as e:
                log.write(f"Error training {task_name}: {str(e)}\n")
                print(f"Error in task {task_name}: {str(e)}")

# latent action space 可視化
def visualize_latent_actions(idm, dataloader, input_dim, task_name="task", save_path=None):
    idm.eval()
    z_list = []

    with torch.no_grad():
        for o_t, o_tp1 in dataloader:
            o_t = o_t.to(device).view(o_t.size(0), -1)
            o_tp1 = o_tp1.to(device).view(o_tp1.size(0), -1)

            z_t = idm(o_t, o_tp1)  # latent action
            z_list.append(z_t.cpu().numpy())

            # 1バッチのみで可視化するなら break もok
            # break

    z_all = np.concatenate(z_list, axis=0)

    # PCAで2次元に圧縮
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_all)

    plt.figure(figsize=(6, 5))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=10, alpha=0.5)
    plt.title(f"Latent Actions (PCA) - {task_name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"latent action image: {save_path}")
    else:
        plt.show()
    plt.close()


# main
if __name__ == "__main__":
    root_path = "datasets"
    #task_dirs = ["cheetah_run", "humanoid_run", "humanoid_walk", "walker_run", "walker_walk"]
    task_dirs = ["cheetah_run", "walker_run", "walker_walk"]

    all_unlabeled = []
    all_labeled = []

    for task in task_dirs:
        video_files = sorted(glob.glob(os.path.join(root_path, task, "train_*_video.mp4")))
        action_files = sorted(glob.glob(os.path.join(root_path, task, "train_*_actions.npz")))

        for vfile, afile in zip(video_files, action_files):
            try:
                unlabeled_ds = VideoActuatorDataset(vfile, None, transform=transform)
                labeled_ds = VideoActuatorDataset(vfile, afile, transform=transform)

                if len(unlabeled_ds) > 0:
                    all_unlabeled.append(unlabeled_ds)
                if len(labeled_ds) > 0:
                    all_labeled.append(labeled_ds)
            except Exception as e:
                print(f"Skipped {vfile}: {e}")

    # loader
    unlabeled_loader = DataLoader(ConcatDataset(all_unlabeled), batch_size=32, shuffle=True)
    labeled_loader = DataLoader(ConcatDataset(all_labeled), batch_size=32, shuffle=True)

    input_dim = 3 * 64 * 64  # 画像サイズ
    action_dim = 6           # 行動次元（アクチュエータの次元数）

    # 設定
    idm, fdm, decoder = train_stage1(
        unlabeled_loader=unlabeled_loader,
        labeled_loader=labeled_loader,
        input_dim=input_dim,
        action_dim=action_dim,
        latent_dim=32,
        NC=100,       # step数
        K=5, # action を入れるタイミング
        task_name="all_tasks"
    )

    # save models
    save_dir = "step_100000_k5/trained_models"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(idm.state_dict(), os.path.join(save_dir, "all_tasks_idm.pt"))
    torch.save(fdm.state_dict(), os.path.join(save_dir, "all_tasks_fdm.pt"))
    torch.save(decoder.state_dict(), os.path.join(save_dir, "all_tasks_decoder.pt"))

    print("All-task training completed and models saved.")

    # latent 可視化
    datasets = load_datasets_from_folder(root_path, transform=transform)
    for task_name, loaders in datasets.items():
        visualize_latent_actions(
            idm=idm,
            dataloader=loaders["unlabeled"],
            input_dim=input_dim,
            task_name=task_name,
            save_path=f"step_100000_k5/latent_z/{task_name}_latent_pca.png"
        )
