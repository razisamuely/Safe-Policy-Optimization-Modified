import torch
import torch.nn as nn

from .util import init


class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions)
        # return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, config=None):
        super(DiagGaussian, self).__init__()
        gain = config["actor_gain"]

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if config is not None:
            self.std_x_coef = config["std_x_coef"]
            self.std_y_coef = config["std_y_coef"]
        else:
            self.std_x_coef = 1.
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def entropy(self):
        return super().entropy()

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, config=None):
        super(Categorical, self).__init__()
        if config is not None:
            gain = config["actor_gain"]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)