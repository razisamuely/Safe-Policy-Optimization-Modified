import numpy as np
from smac_wrapper import SMACShareEnv

def test_collision_cost():
    print("Testing Collision Cost in MACPO Wrapper...")
    env = SMACShareEnv(map_name="8m", cost_type="collision")
    obs, share_obs, avail = env.reset()
    
    print(f"Environment initialized. Agents: {env.num_agents}")
    
    # Run random actions and check for costs
    for i in range(100):
        # Sample random actions (0 is no-op, 2-5 are move)
        actions = [np.random.randint(0, env.n_actions) for _ in range(env.num_agents)]
        obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)
        
        step_cost = sum([c[0] for c in costs])
        if step_cost > 0:
            print(f"Step {i}: SUCCESS! Found collision cost: {step_cost}")
            env.close()
            return True
            
    print("Failed to find collisions in 100 random steps.")
    env.close()
    return False

if __name__ == "__main__":
    test_collision_cost()
