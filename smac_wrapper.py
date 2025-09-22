import numpy as np
from gymnasium.spaces import Box, Discrete
from smac.env import StarCraft2Env

class SMACShareEnv:
    def __init__(self, map_name="3m", cost_type=None, cost_threshold=0.5, **kwargs):
        self.env = StarCraft2Env(map_name=map_name, **kwargs)
        self.map_name = map_name
        self.cost_type = cost_type
        self.cost_threshold = cost_threshold
        
        env_info = self.env.get_env_info()
        self.num_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.obs_size = env_info["obs_shape"]
        self.state_size = env_info["state_shape"]
        self.share_obs_size = self.state_size + self.num_agents
        
        self.observation_spaces = {}
        self.share_observation_spaces = {}
        self.action_spaces = {}
        
        for agent in range(self.num_agents):
            self.observation_spaces[f"agent_{agent}"] = Box(
                low=-10.0, high=10.0, shape=(self.obs_size + self.num_agents,)
            )
            self.share_observation_spaces[f"agent_{agent}"] = Box(
                low=-10.0, high=10.0, shape=(self.share_obs_size,)
            )
            # Use discrete action space for SMAC
            self.action_spaces[f"agent_{agent}"] = Discrete(self.n_actions)
        
        self.prev_health = None
        self.prev_state = None
        self.prev_kills = 0

    def _get_obs(self):
        obs = self.env.get_obs()
        obs_n = []
        for a in range(self.num_agents):
            agent_id_feats = np.zeros(self.num_agents, dtype=np.float32)
            agent_id_feats[a] = 1.0
            if obs[a] is not None:
                obs_i = np.concatenate([obs[a], agent_id_feats])
                obs_i = (obs_i - np.mean(obs_i)) / (np.std(obs_i) + 1e-8)

            else:
                obs_i = np.zeros(self.obs_size + self.num_agents, dtype=np.float32)
            obs_n.append(obs_i)
        return obs_n

    def _get_share_obs(self):
        state = self.env.get_state()
        if state is None:
            state = np.zeros(self.state_size, dtype=np.float32)
        
        agent_alive = np.zeros(self.num_agents, dtype=np.float32)
        for i in range(self.num_agents):
            try:
                unit = self.env.get_unit_by_id(i)
                agent_alive[i] = 1.0 if unit and unit.health > 0 else 0.0
            except:
                agent_alive[i] = 0.0
        
        share_state = np.concatenate([state, agent_alive])
        share_state = share_state / (np.linalg.norm(share_state) + 1e-8)
        
        share_obs = []
        for _ in range(self.num_agents):
            share_obs.append(share_state.copy())
        return share_obs

    def _get_avail_actions(self):
        avail_actions = self.env.get_avail_actions()
        avail_actions_padded = np.zeros((self.num_agents, self.n_actions), dtype=np.float32)
        for i in range(self.num_agents):
            if i < len(avail_actions):
                avail_actions_padded[i][:len(avail_actions[i])] = avail_actions[i]
        return avail_actions_padded

    def _compute_costs(self, reward, terminated):
        costs = np.zeros(self.num_agents, dtype=np.float32)
        
        try:
            info = getattr(self, '_current_info', {})
            team_cost = self.get_cost(info)
            costs.fill(team_cost)
        except Exception as e:
            raise Exception(f"Error computing cost '{self.cost_type}': {e}")

        return costs

    def _convert_actions_with_masking(self, actions):
        actions_array = []
        avail_actions = self.env.get_avail_actions()
        
        for agent_id in range(self.num_agents):
            action = actions[agent_id]
            if hasattr(action, 'item'):
                action = action.item()
            elif hasattr(action, '__len__') and len(action) == 1:
                action = action[0]
            action = int(action)

            if agent_id < len(avail_actions):
                avail = avail_actions[agent_id]
                if avail[action] == 0:
                    # If policy chose unavailable action, pick first available
                    valid_actions = np.where(avail)[0]
                    if len(valid_actions) > 0:
                        action = valid_actions[0]
                    else:
                        action = 0  # fallback no-op
            else:
                action = 0  # agent doesn’t exist, use no-op

            actions_array.append(action)

        return actions_array


    def reset(self, seed=None):
        self.env.reset()
        self.prev_health = None
        self.prev_state = None
        self.prev_kills = 0
        return self._get_obs(), self._get_share_obs(), self._get_avail_actions()

    def step(self, actions):
        # Convert actions with proper masking
        actions_array = self._convert_actions_with_masking(actions)
        
        try:
            reward, terminated, info = self.env.step(actions_array)
        except Exception as e:
            print(f"Error in SMAC step: {e}")
            # Return safe defaults if step fails
            reward = 0
            terminated = True
            info = {}
        
        if terminated:
            self.env.reset()
            self.prev_health = None
            self.prev_state = None
            self.prev_kills = 0
        # Store info for advanced cost computation
        self._current_info = info
        costs = self._compute_costs(reward, terminated)
        
        obs = self._get_obs()
        share_obs = self._get_share_obs()
        avail_actions = self._get_avail_actions()
        
        rewards = [[float(reward)] for _ in range(self.num_agents)]  # <-- Add float() conversion
        cost_list = [[float(cost)] for cost in costs]  # <-- Add float() conversion
        dones = [terminated] * self.num_agents
        infos = [info] * self.num_agents
        
        return obs, share_obs, rewards, cost_list, dones, infos, avail_actions

    def close(self):
        try:
            self.env.close()
        except:
            pass

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def get_env_info(self):
        return self.env.get_env_info()

    def get_health(self, agent_id):
        """Get health of specific agent"""
        unit = self.env.get_unit_by_id(agent_id)
        return unit.health if unit else 0

    def get_agent_position(self, agent_id):
        """Get position coordinates of agent"""
        unit = self.env.get_unit_by_id(agent_id)
        if unit:
            return (unit.pos.x, unit.pos.y)
        else:
            raise ValueError(f"Agent {agent_id} not found in the environment.")

    def get_enemy_distances(self, agent_id):
        """Calculate distances to all enemies for given agent"""
        agent_pos = self.get_agent_position(agent_id)

        distances = []
        enemies = self.env.enemies
        for enemy in enemies.values():
            if enemy.health > 0:  # Only alive enemies
                enemy_pos = (enemy.pos.x, enemy.pos.y)
                dist = np.sqrt((agent_pos[0] - enemy_pos[0]) ** 2 + (agent_pos[1] - enemy_pos[1]) ** 2)
                distances.append(dist)
        return distances

    def is_in_danger_zone(self, pos):
        """Check if position is in predefined danger zone"""
        if not pos:
            return False
        # Define danger zone as center area (customize based on map)
        center_x, center_y = 16, 16  # Adjust based on map size
        danger_radius = 8
        dist_to_center = np.sqrt((pos[0] - center_x) ** 2 + (pos[1] - center_y) ** 2)
        return dist_to_center < danger_radius

    def calculate_formation_spread(self, positions):
        """Calculate how spread out the formation is"""
        valid_positions = [pos for pos in positions if pos is not None]
        if len(valid_positions) < 2:
            return 0

        # Calculate average pairwise distance
        total_distance = 0
        pairs = 0
        for i in range(len(valid_positions)):
            for j in range(i + 1, len(valid_positions)):
                pos1, pos2 = valid_positions[i], valid_positions[j]
                dist = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
                total_distance += dist
                pairs += 1

        return total_distance / pairs if pairs > 0 else 0

    def get_cost_resource_waste(self, info):
        # Get actual action indices from one-hot encoded actions
        if hasattr(self.env, "last_action") and self.env.last_action is not None:
            action_indices = [row.argmax() for row in self.env.last_action]
            # Count attack actions (actions >= n_actions_no_attack are attacks)
            attack_count = sum(1 for action in action_indices if action >= self.env.n_actions_no_attack)
        else:
            attack_count = 0

        # Track kills this step
        current_kills = info.get("dead_enemies", 0)
        kills_this_step = current_kills - getattr(self, "prev_kills", 0)
        self.prev_kills = current_kills

        # Cost = attacks that didn't result in kills (wasted shots)
        wasted_shots = max(0, attack_count - kills_this_step)
        return wasted_shots

    def reward_shaping(self, reward, info, ratio):
        """
        Apply reward shaping based on cost.
        The cost is subtracted from the reward to encourage efficiency.
        """
        cost = info.get("cost", {}).get(0, 0)
        if cost > 0:
            return reward - ratio * cost
        return reward

    def get_cost_aggressive_positioning(self, info):
        """Cost based on risky positioning (too close to enemies)"""
        total_risk = 0
        for agent_id in range(self.num_agents):
            if self.get_health(agent_id) > 0:  # Only alive agents
                enemy_distances = self.get_enemy_distances(agent_id)
                # Count how many enemies are within unsafe distance
                close_enemies = sum(1 for d in enemy_distances if d < getattr(self, 'safe_distance', 3.0))
                total_risk += close_enemies
        return total_risk

    def get_cost_danger_zone_violation(self, info):
        """Cost based on entering dangerous areas"""
        violations = 0
        for agent_id in range(self.num_agents):
            if self.get_health(agent_id) > 0:  # Only alive agents
                pos = self.get_agent_position(agent_id)
                if self.is_in_danger_zone(pos):
                    violations += 1
        return violations

    def get_cost_formation_breaking(self, info):
        """Cost based on breaking formation (agents too spread out)"""
        positions = []
        for agent_id in range(self.num_agents):
            if self.get_health(agent_id) > 0:  # Only alive agents
                pos = self.get_agent_position(agent_id)
                if pos:
                    positions.append(pos)

        if len(positions) < 2:
            return 0

        formation_spread = self.calculate_formation_spread(positions)
        # Return cost if formation is too spread out
        return max(0, formation_spread - getattr(self, 'formation_threshold', 5.0))

    def get_cost_combined(self, info):
        """Combined cost function using multiple factors"""
        resource_cost = self.get_cost_resource_waste(info) * 0.3
        position_cost = self.get_cost_aggressive_positioning(info) * 0.3
        danger_cost = self.get_cost_danger_zone_violation(info) * 0.2
        formation_cost = self.get_cost_formation_breaking(info) * 0.2

        return resource_cost + position_cost + danger_cost + formation_cost

    def get_cost_debug_constant(self, info):
        """Cost based on a constant value for debugging purposes"""
        return 1.0

    def get_cost(self, info):
        """Main cost function - selects based on cost_type"""
        if self.cost_type == "resource_waste":
            return self.get_cost_resource_waste(info)
        elif self.cost_type == "aggressive_positioning":
            return self.get_cost_aggressive_positioning(info)
        elif self.cost_type == "danger_zone":
            return self.get_cost_danger_zone_violation(info)
        elif self.cost_type == "formation_breaking":
            return self.get_cost_formation_breaking(info)
        elif self.cost_type == "combined":
            return self.get_cost_combined(info)
        elif self.cost_type == "dead_allies":
            return info.get("dead_allies", 0)
        elif self.cost_type == "damage":
            return self.get_cost_damage()
        elif self.cost_type == "death":
            return self.get_cost_death()
        elif self.cost_type == "proximity":
            return self.get_cost_proximity()
        elif self.cost_type == "debug_constant":
            return self.get_cost_debug_constant(info)
        else:
            raise ValueError(f"Unknown cost_type: {self.cost_type}")

    def get_cost_damage(self):
        """Cost based on damage taken by agents"""
        total_damage = 0
        current_health = []
        for i in range(self.num_agents):
            unit = self.env.get_unit_by_id(i)
            if unit and unit.health > 0:
                max_health = unit.health_max + (unit.shield_max if hasattr(unit, 'shield_max') else 0)
                current_health = unit.health + (unit.shield if hasattr(unit, 'shield') else 0)
                damage_taken = max_health - current_health
                total_damage += damage_taken

        return total_damage

    def get_cost_death(self):
        """Cost based on agent deaths"""
        total_cost = 0
        for i in range(self.num_agents):
            try:
                unit = self.env.get_unit_by_id(i)
                if not unit or unit.health <= 0:
                    total_cost += 1.0
            except:
                total_cost += 1.0
        return total_cost

    def get_cost_proximity(self):
        """Cost based on proximity to enemies"""
        total_cost = 0
        try:
            enemy_units = self.env.enemies
            for i in range(self.num_agents):
                unit = self.env.get_unit_by_id(i)
                if unit and unit.health > 0:
                    for enemy in enemy_units.values():
                        if enemy.health > 0:
                            dist = self.env.distance(unit.pos.x, unit.pos.y, enemy.pos.x, enemy.pos.y)
                            if dist < self.cost_threshold:
                                total_cost += 1.0
                                break
        except:
            pass
        return total_cost