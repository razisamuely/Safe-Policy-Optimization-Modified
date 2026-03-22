#!/usr/bin/env python3
"""Test script for get_cost_dead_allies_incremental"""

import random
import numpy as np

class MockInfo:
    def __init__(self, dead_allies, battle_won=False):
        self.dead_allies = dead_allies
        self.battle_won = battle_won
    
    def get(self, key, default):
        if key == "dead_allies":
            return self.dead_allies
        elif key == "battle_won":
            return self.battle_won
        return default

class MockEnv:
    def __init__(self, num_agents=8):
        self.num_agents = num_agents

class CostTester:
    def __init__(self, num_agents=8):
        self.num_agents = num_agents
        self.prev_deaths = 0
    
    def get_cost_dead_allies_incremental(self, info, terminated):
        """Cost based on NEW deaths this step only"""
        current_deaths = info.get("dead_allies", 0)
        new_deaths = current_deaths - getattr(self, "prev_deaths", 0)
        new_deaths = max(0, new_deaths)
        self.prev_deaths = max(0, current_deaths)
        
        # If episode ended losing, count survivors as dead
        if terminated:
            battle_won = info.get("battle_won", False)
            
            if not battle_won:
                # Return only survivors (what hasn't been counted yet)
                total_allies = self.num_agents
                survivors = total_allies - current_deaths
                if survivors > 0:
                    new_deaths = survivors
                    self.prev_deaths = total_allies
                    return new_deaths
                # All dead already, no survivors penalty needed
                self.prev_deaths = total_allies
                return 0
            # Battle won, reset normally
            self.prev_deaths = 0
        
        return new_deaths
    
    def reset(self):
        self.prev_deaths = 0

def test_episode(tester, episode_num, max_steps=18):
    """Simulate one episode"""
    tester.reset()
    total_cost = 0
    current_deaths = 0
    steps = []
    
    # Random number of steps (short episode)
    num_steps = random.randint(2, max_steps)
    
    for step in range(num_steps):
        # Randomly kill 0-2 allies this step (if not all dead)
        if current_deaths < tester.num_agents:
            new_deaths_this_step = random.randint(0, min(2, tester.num_agents - current_deaths))
            current_deaths += new_deaths_this_step
        else:
            new_deaths_this_step = 0
        
        # Last step: episode terminates
        terminated = (step == num_steps - 1)
        battle_won = random.random() < 0.3  # 30% win rate
        
        info = MockInfo(current_deaths, battle_won)
        cost = tester.get_cost_dead_allies_incremental(info, terminated)
        total_cost += cost
        
        steps.append({
            'step': step,
            'current_deaths': current_deaths,
            'cost': cost,
            'terminated': terminated,
            'battle_won': battle_won
        })
    
    return {
        'episode': episode_num,
        'total_cost': total_cost,
        'final_deaths': current_deaths,
        'battle_won': battle_won,
        'steps': steps
    }

def main():
    num_agents = 8
    num_episodes = 10000000
    
    tester = CostTester(num_agents)
    results = []
    
    print(f"Testing {num_episodes} episodes with {num_agents} allies...")
    print("-" * 60)
    
    violations = []
    for ep in range(num_episodes):
        result = test_episode(tester, ep, max_steps=18)
        results.append(result)
        
        if result['total_cost'] > num_agents:
            violations.append(result)
            if len(violations) <= 10:  # Print first 10 violations
                print(f"Ep {ep} [✗] Cost: {result['total_cost']:.0f}, "
                      f"Deaths: {result['final_deaths']}, Battle: {'WON' if result['battle_won'] else 'LOST'}")
        
        if (ep + 1) % 10000 == 0:
            print(f"Progress: {ep + 1}/{num_episodes} episodes...")
    
    print("-" * 60)
    
    # Summary
    max_cost = max(r['total_cost'] for r in results)
    avg_cost = np.mean([r['total_cost'] for r in results])
    losses = [r for r in results if not r['battle_won']]
    losses_cost = [r['total_cost'] for r in losses]
    
    print(f"Max cost: {max_cost:.2f}")
    print(f"Avg cost: {avg_cost:.2f}")
    print(f"Losses: {len(losses)}/{num_episodes}, Avg cost on loss: {np.mean(losses_cost):.2f}")
    
    # Check for violations
    if violations:
        print(f"\n❌ FAILED: {len(violations)} episodes with cost > {num_agents}")
        for v in violations[:5]:  # Show first 5
            print(f"  Episode {v['episode']}: cost={v['total_cost']:.0f}, deaths={v['final_deaths']}, "
                  f"battle={'WON' if v['battle_won'] else 'LOST'}")
    else:
        print(f"\n✅ PASSED: All {num_episodes} episodes have cost <= {num_agents}")

if __name__ == "__main__":
    main()

