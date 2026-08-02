import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import gradio as gr


# =========================
# CẤU HÌNH
# =========================

BASE_DIR = r"C:\Users\Admin\Desktop\cs106\doannhom\TD3-master\models"

DEFAULT_SEED = 0

METHODS = {
    "TD3": {
        "folder": "td3",
        "prefix": "TD3"
    },
    "DDPG": {
        "folder": "ddpg",
        "prefix": "DDPG"
    },
    "OurDDPG": {
        "folder": "ourddpg",
        "prefix": "OurDDPG"
    }
}


# =========================
# ACTOR 256-256
# Dùng cho TD3 và OurDDPG
# =========================

class Actor256(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor256, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        return self.max_action * torch.tanh(self.l3(a))


# =========================
# ACTOR 400-300
# Dùng cho DDPG
# =========================

class Actor400(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor400, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        return self.max_action * torch.tanh(self.l3(a))


# =========================
# CHỌN ACTOR THEO PHƯƠNG PHÁP
# =========================

def create_actor(method_name, env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if method_name == "DDPG":
        return Actor400(state_dim, action_dim, max_action)

    elif method_name == "TD3":
        return Actor256(state_dim, action_dim, max_action)

    elif method_name == "OurDDPG":
        return Actor256(state_dim, action_dim, max_action)

    else:
        raise ValueError(f"Không biết phương pháp: {method_name}")


# =========================
# QUÉT DANH SÁCH BÀI TOÁN
# =========================

def scan_envs():
    envs = set()

    for method_name, cfg in METHODS.items():
        folder = os.path.join(BASE_DIR, cfg["folder"])
        prefix = cfg["prefix"]

        if not os.path.exists(folder):
            print("Không tìm thấy thư mục:", folder)
            continue

        for file_name in os.listdir(folder):
            if file_name.startswith(prefix) and file_name.endswith("_actor"):
                # VD: TD3_Ant-v5_0_actor
                name = file_name.replace(f"{prefix}_", "")
                name = name.replace("_actor", "")

                # VD: Ant-v5_0
                parts = name.rsplit("_", 1)

                if len(parts) == 2:
                    env_name, seed = parts

                    if seed.isdigit():
                        envs.add(env_name)

    return sorted(list(envs))


# =========================
# LẤY ĐƯỜNG DẪN FILE ACTOR
# =========================

def get_actor_path(method_name, env_name, seed=DEFAULT_SEED):
    cfg = METHODS[method_name]

    actor_path = os.path.join(
        BASE_DIR,
        cfg["folder"],
        f"{cfg['prefix']}_{env_name}_{seed}_actor"
    )

    return actor_path


# =========================
# LOAD POLICY
# =========================

def load_policy(method_name, env_name, seed=DEFAULT_SEED):
    env = gym.make(env_name, render_mode="rgb_array")

    actor_path = get_actor_path(method_name, env_name, seed)

    if not os.path.exists(actor_path):
        env.close()
        raise FileNotFoundError(f"Không tìm thấy file actor: {actor_path}")

    actor = create_actor(method_name, env)

    state_dict = torch.load(actor_path, map_location="cpu")
    actor.load_state_dict(state_dict)
    actor.eval()

    print(f"Loaded {method_name}: {actor_path}")

    return actor, env


# =========================
# CHỌN ACTION
# =========================

def select_action(actor, state):
    state = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        action = actor(state).cpu().numpy()[0]

    return action


# =========================
# THÊM TÊN LÊN FRAME
# =========================

def add_title(frame, title):
    frame = frame.copy()

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 45),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        title,
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    return frame


# =========================
# GHÉP 3 FRAME
# =========================

def combine_frames(frames):
    target_w = 480
    target_h = 360

    resized_frames = []

    for frame in frames:
        frame = cv2.resize(frame, (target_w, target_h))
        resized_frames.append(frame)

    combined = np.concatenate(resized_frames, axis=1)

    return combined


# =========================
# CHẠY DEMO
# =========================

def run_compare_demo(env_name, max_steps):
    seed = DEFAULT_SEED

    methods = ["TD3", "DDPG", "OurDDPG"]

    actors = {}
    envs = {}
    states = {}
    dones = {}

    try:
        for method in methods:
            actor, env = load_policy(method, env_name, seed)

            state, info = env.reset(seed=seed)

            actors[method] = actor
            envs[method] = env
            states[method] = state
            dones[method] = False

        frames_output = []

        for step in range(int(max_steps)):
            panel_frames = []

            for method in methods:
                env = envs[method]
                actor = actors[method]

                if not dones[method]:
                    action = select_action(actor, states[method])

                    next_state, reward, terminated, truncated, info = env.step(action)

                    states[method] = next_state
                    dones[method] = terminated or truncated

                frame = env.render()
                frame = add_title(frame, method)

                panel_frames.append(frame)

            combined = combine_frames(panel_frames)
            frames_output.append(combined)

            if all(dones.values()):
                break

        if len(frames_output) == 0:
            raise RuntimeError("Không render được frame nào.")

        video_path = os.path.abspath(f"compare_{env_name}.mp4")

        h, w, _ = frames_output[0].shape

        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (w, h)
        )

        for frame in frames_output:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()

        print("Đã lưu video tại:", video_path)

        return video_path

    finally:
        for env in envs.values():
            env.close()


# =========================
# GIAO DIỆN GRADIO
# =========================

env_list = scan_envs()

if len(env_list) == 0:
    print("Không tìm thấy bài toán nào.")
    print("Kiểm tra lại BASE_DIR và tên file actor.")


with gr.Blocks() as demo:
    gr.Markdown("# Demo so sánh TD3 / DDPG / OurDDPG")

    with gr.Row():
        env_dropdown = gr.Dropdown(
            choices=env_list,
            label="Chọn bài toán",
            value=env_list[0] if len(env_list) > 0 else None
        )

        max_steps_slider = gr.Slider(
            minimum=100,
            maximum=2000,
            value=500,
            step=100,
            label="Số bước render tối đa"
        )

    run_button = gr.Button("Chạy demo")

    video_output = gr.Video(
        label="Render so sánh 3 phương pháp"
    )

    run_button.click(
        fn=run_compare_demo,
        inputs=[env_dropdown, max_steps_slider],
        outputs=video_output
    )


if __name__ == "__main__":
    demo.launch(show_error=True)