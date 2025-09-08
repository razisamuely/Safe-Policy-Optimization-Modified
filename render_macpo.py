import copy
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import time
import json
import argparse
import imageio
from PIL import Image, ImageDraw, ImageFont

from safepo.common.env import make_ma_smac_env
from safepo.common.popart import PopArt
from safepo.common.model import MultiAgentActor as Actor, MultiAgentCritic as Critic
from safepo.utils.config import set_np_formatting, set_seed, smac_map


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


class MACPO_Policy():
    def __init__(self, config, obs_space, cent_obs_space, act_space):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.share_obs_space = cent_obs_space

        self.actor = Actor(config, obs_space, act_space, self.config["device"])
        self.critic = Critic(config, cent_obs_space, self.config["device"])
        self.cost_critic = Critic(config, cent_obs_space, self.config["device"])

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor


def create_episode_title_frame(episode_num, total_episodes, width=640, height=480, duration_frames=30):
    """Create a title frame for episode transitions"""
    # Create a black background
    img = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        try:
            font_large = ImageFont.truetype("arial.ttf", 48)
            font_small = ImageFont.truetype("arial.ttf", 24)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # Main title
    title_text = f"Episode {episode_num}"
    subtitle_text = f"of {total_episodes}"
    
    # Calculate text positions (center the text)
    title_bbox = draw.textbbox((0, 0), title_text, font=font_large)
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]
    
    subtitle_bbox = draw.textbbox((0, 0), subtitle_text, font=font_small)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    subtitle_height = subtitle_bbox[3] - subtitle_bbox[1]
    
    # Draw the text
    title_x = (width - title_width) // 2
    title_y = (height - title_height) // 2 - 20
    
    subtitle_x = (width - subtitle_width) // 2
    subtitle_y = title_y + title_height + 10
    
    draw.text((title_x, title_y), title_text, fill='white', font=font_large)
    draw.text((subtitle_x, subtitle_y), subtitle_text, fill='gray', font=font_small)
    
    # Convert PIL image to numpy array
    frame = np.array(img)
    
    # Return multiple frames for the desired duration
    return [frame] * duration_frames


class MACPO_Renderer:
    def __init__(self, model_dir, config_path, map_name="3m", cost_type="debug_constant", save_gifs=True, gif_dir="rendered_gifs"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model_dir = model_dir
        self.map_name = map_name
        self.cost_type = cost_type
        self.save_gifs = save_gifs
        
        if self.save_gifs:
            self.gif_dir = gif_dir
            os.makedirs(self.gif_dir, exist_ok=True)
            print(f"GIFs will be saved to: {self.gif_dir}")
        
        self.config["n_rollout_threads"] = 1
        self.config["n_eval_rollout_threads"] = 1
        
        if self.config.get("device", "cpu") not in ["cpu", "cuda"]:
            self.config["device"] = "cpu"
        
        self.env = make_ma_smac_env(
            map_name=self.map_name,
            cost_type=self.cost_type,
            seed=self.config["seed"],
            cfg_train=self.config
        )
        
        self.num_agents = self.env.num_agents
        
        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.env.share_observation_space[agent_id]
            po = MACPO_Policy(
                self.config,
                self.env.observation_space[agent_id],
                share_observation_space,
                self.env.action_space[agent_id]
            )
            self.policy.append(po)
        
        self.load_models()
        
        for agent_id in range(self.num_agents):
            self.policy[agent_id].actor.eval()
            self.policy[agent_id].critic.eval()
            self.policy[agent_id].cost_critic.eval()

    def load_models(self):
        for agent_id in range(self.num_agents):
            actor_path = os.path.join(self.model_dir, f"actor_agent{agent_id}.pt")
            if os.path.exists(actor_path):
                actor_state_dict = torch.load(actor_path, map_location=self.config["device"])
                self.policy[agent_id].actor.load_state_dict(actor_state_dict)
                print(f"Loaded actor for agent {agent_id}")
            else:
                print(f"Warning: Actor model not found for agent {agent_id} at {actor_path}")
            
            critic_path = os.path.join(self.model_dir, f"critic_agent{agent_id}.pt")
            if os.path.exists(critic_path):
                critic_state_dict = torch.load(critic_path, map_location=self.config["device"])
                self.policy[agent_id].critic.load_state_dict(critic_state_dict)
                print(f"Loaded critic for agent {agent_id}")
            else:
                print(f"Warning: Critic model not found for agent {agent_id} at {critic_path}")

    @torch.no_grad()
    def render_episodes(self, num_episodes=5, deterministic=True, fps=10):
        
        all_frames = []
        frame_width, frame_height = 640, 480  # Default size, will be updated
        
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
            
            episode_frames = []
            
            obs, share_obs, available_actions = self.env.reset()
            
            rnn_states = torch.zeros(
                self.config["n_eval_rollout_threads"], 
                self.num_agents, 
                self.config["recurrent_N"], 
                self.config["hidden_size"],
                device=self.config["device"]
            )
            masks = torch.ones(
                self.config["n_eval_rollout_threads"], 
                self.num_agents, 
                1, 
                device=self.config["device"]
            )
            
            episode_reward = 0.0
            episode_cost = 0.0
            step = 0
            
            while True:
                frame = self.env.render(mode="rgb_array")
                
                if self.save_gifs and frame is not None:
                    # Get frame dimensions for title screens (only once)
                    if episode == 0 and step == 0:
                        frame_height, frame_width = frame.shape[:2]
                        print(f"Detected frame size: {frame_width}x{frame_height}")
                        
                        # Add initial title screen with correct dimensions
                        print(f"Adding initial title screen...")
                        initial_title_frames = create_episode_title_frame(
                            1, 
                            num_episodes, 
                            width=frame_width, 
                            height=frame_height, 
                            duration_frames=fps  # 1 second duration at given fps
                        )
                        all_frames.extend(initial_title_frames)
                    
                    episode_frames.append(frame)
                    all_frames.append(frame)
                
                actions_collector = []
                
                for agent_id in range(self.num_agents):
                    if 'Frank' in self.config['env_name']:
                        obs_to_eval = obs[agent_id]
                    else:
                        obs_to_eval = obs[:, agent_id]
                    
                    action, temp_rnn_state = self.policy[agent_id].act(
                        obs_to_eval,
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        available_actions=available_actions[:, agent_id] if available_actions is not None else None,
                        deterministic=deterministic
                    )
                    
                    rnn_states[:, agent_id] = temp_rnn_state
                    actions_collector.append(action)
                
                if self.config["env_name"] == "Safety9|8HumanoidVelocity-v0":
                    zeros = torch.zeros(actions_collector[-1].shape[0], 1)
                    actions_collector[-1] = torch.cat((actions_collector[-1], zeros), dim=1)
                
                obs, share_obs, rewards, costs, dones, infos, available_actions = self.env.step(actions_collector)
                
                dones_env = torch.all(dones, dim=1)
                reward_env = torch.mean(rewards, dim=1).flatten()
                cost_env = torch.mean(costs, dim=1).flatten()
                
                episode_reward += reward_env.item()
                episode_cost += cost_env.item()
                step += 1
                
                print(f"Step {step}: Reward = {reward_env.item():.3f}, Cost = {cost_env.item():.3f}")
                
                if dones_env.item():
                    print(f"Episode {episode + 1} finished after {step} steps")
                    print(f"Total Reward: {episode_reward:.3f}")
                    print(f"Total Cost: {episode_cost:.3f}")
                    
                    if self.save_gifs and episode_frames:
                        gif_path = os.path.join(
                            self.gif_dir, 
                            f"episode_{episode + 1}_reward_{episode_reward:.1f}_cost_{episode_cost:.1f}.gif"
                        )
                        imageio.mimsave(gif_path, episode_frames, fps=fps)
                        print(f"Saved episode GIF: {gif_path}")
                    
                    # Add title screen for next episode (if not the last episode)
                    if episode < num_episodes - 1:
                        print(f"Adding title screen for Episode {episode + 2}")
                        title_frames = create_episode_title_frame(
                            episode + 2, 
                            num_episodes, 
                            width=frame_width, 
                            height=frame_height, 
                            duration_frames=fps  # 1 second duration at given fps
                        )
                        all_frames.extend(title_frames)
                    
                    break
                
                rnn_states[dones_env == True] = torch.zeros(
                    (dones_env == True).sum(), 
                    self.num_agents, 
                    self.config["recurrent_N"], 
                    self.config["hidden_size"], 
                    device=self.config["device"]
                )
                
                masks = torch.ones(
                    self.config["n_eval_rollout_threads"], 
                    self.num_agents, 
                    1, 
                    device=self.config["device"]
                )
                masks[dones_env == True] = torch.zeros(
                    (dones_env == True).sum(), 
                    self.num_agents, 
                    1,
                    device=self.config["device"]
                )
        
        if self.save_gifs and all_frames:
            gif_path = os.path.join(self.gif_dir, f"all_{num_episodes}_episodes.gif")
            imageio.mimsave(gif_path, all_frames, fps=fps)
            print(f"Saved combined GIF: {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="Render MACPO trained model")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--config-path", type=str, help="Path to config.json")
    parser.add_argument("--map-name", type=str, default="3m", help="SMAC map name")
    parser.add_argument("--cost-type", type=str, default="damage", help="Cost type")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to render")
    parser.add_argument("--deterministic", type=bool, default=True, help="Use deterministic actions")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save-gifs", type=bool, default=True, help="Save episodes as GIFs")
    parser.add_argument("--gif-dir", type=str, default="rendered_gifs", help="Directory to save GIFs")
    parser.add_argument("--fps", type=int, default=10, help="GIF FPS")
    
    args = parser.parse_args()
    
    if args.config_path is None:
        args.config_path = os.path.join(os.path.dirname(args.model_dir), "config.json")
    
    set_np_formatting()
    set_seed(args.seed)
    
    print(f"Loading model from: {args.model_dir}")
    print(f"Using config from: {args.config_path}")
    print(f"Map: {args.map_name}, Cost type: {args.cost_type}")
    print(f"Rendering {args.num_episodes} episodes...")
    if args.save_gifs:
        print(f"GIFs will be saved to: {args.gif_dir}")
    
    renderer = MACPO_Renderer(
        model_dir=args.model_dir,
        config_path=args.config_path,
        map_name=args.map_name,
        cost_type=args.cost_type,
        save_gifs=args.save_gifs,
        gif_dir=args.gif_dir
    )
    
    renderer.render_episodes(
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        fps=args.fps
    )
    
    print("\nRendering completed!")
    if args.save_gifs:
        print(f"Check the '{args.gif_dir}' directory for your GIFs!")


if __name__ == '__main__':
    main()

# python render_macpo.py --model-dir <model_dir> --config-path <config_path> --map-name <map_name> --num-episodes <num_episodes> --deterministic <deterministic> --save-gifs <save_gifs> --gif-dir <gif_dir> --fps <fps>
# #   --map-name 8m --model-dir safepo/runs/debug/8m/macpo/seed-000-2025-09*/models --save-video --output-file policy_8m.gif --num-episodes 3 --mode rgb_array --deterministic
# safepo/runs/Base/8m/macpo/seed-000-2025-09-01-06-38-11
# python render_macpo.py --model-dir safepo/runs/Base/8m/macpo/seed-000-2025-09-01-06-38-11/models_seed0 --config-path safepo/runs/Base/8m/macpo/seed-000-2025-09-01-06-38-11/config.json --map-name 8m --num-episodes 3 --deterministic True --save-gifs True --gif-dir rendered_gifs --fps 10
# python render_macpo.py --model-dir safepo/runs/Base/3m/macpo/seed-000-2025-08-30-14-36-49/models_seed0 --config-path safepo/runs/Base/3m/macpo/seed-000-2025-08-30-14-36-49/config.json --map-name 3m --num-episodes 3 --deterministic True --save-gifs True --gif-dir rendered_gifs --fps 10