# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import torch.nn as nn
import torch
from safepo.utils.distributions import DiagGaussian, Categorical



    
class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()
        self.action_type = action_space.__class__.__name__
        self.mixed_action = False
        self.multi_discrete = False
        if self.action_type == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain, args)
        else:
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain, args)
    
    def forward(self, x, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        action_logits = self.action_out(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample() 
        action_log_probs = action_logits.log_probs(actions)
        
        return actions, action_log_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_logits = self.action_out(x, available_actions)
        action_log_probs = action_logits.log_probs(action)
        if active_masks is not None:
            dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy

    def evaluate_actions_trpo(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_logits = self.action_out(x, available_actions)
        
        # Check for NaN values in the logits and handle them
        if self.action_type == "Discrete" and hasattr(action_logits, 'logits'):
            if torch.isnan(action_logits.logits).any():
                print(f"Warning: NaN detected in action logits, shape: {action_logits.logits.shape}")
                print(f"Input x stats - mean: {x.mean():.6f}, std: {x.std():.6f}, min: {x.min():.6f}, max: {x.max():.6f}")
                print(f"Input x has NaN: {torch.isnan(x).any()}")
                # Create new logits with small random values
                new_logits = torch.randn_like(action_logits.logits) * 0.01
                action_logits = type(action_logits)(logits=new_logits)
                print("Reinitialized action_logits to small random values")
    
        if self.action_type == "Discrete":
            action_mu = None
            action_std = None
        else:
            action_mu = action_logits.mean
            action_std = action_logits.stddev
            
        action_log_probs = action_logits.log_probs(action)
        all_probs = None
        if active_masks is not None:
            dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy, action_mu, action_std, all_probs
