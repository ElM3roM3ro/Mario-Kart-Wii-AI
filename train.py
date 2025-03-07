from env import MarioKartDolphinEnv
from agent import RainbowDQN

def train(num_episodes=10):
    env = MarioKartDolphinEnv(target_width=128, target_height=128)
    agent = RainbowDQN(
        state_shape=(4,128,128),   # 4 grayscale frames
        num_actions=env.action_space_size,
        lr=1e-4,
        buffer_size=50000,
        batch_size=32,
        gamma=0.99,
        v_min=-10.0,
        v_max=10.0,
        num_atoms=51,
        target_update_interval=1000,
        device='cuda'  # if you have GPU
    )

    max_steps_per_episode = 2000

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            # Agent picks an action
            action_idx = agent.select_action(obs)

            # Step environment
            next_obs, reward, done, info = env.step(action_idx)

            # Store transition
            agent.store_transition((obs, action_idx, reward, next_obs, done))

            # Update agent
            agent.update()
            agent.frame_count += 1

            obs = next_obs
            total_reward += reward
            steps += 1

        print(f"Episode {episode+1}, total_reward={total_reward}")


if __name__ == "__main__":
    train(num_episodes=10)
