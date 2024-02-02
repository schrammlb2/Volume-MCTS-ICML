import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam

from typing import Dict, Union
from abc import abstractmethod


class Loss(nn.Module):
    """ABC for the loss classes."""

    @abstractmethod
    def forward(self):
        ...

    @abstractmethod
    def _calculate_policy_loss(self) -> torch.Tensor:
        ...

    @abstractmethod
    def _calculate_value_loss(
        self, V_hat: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        ...


class AlphaZeroLoss(Loss):
    """AlphaZero loss.

    This class implements the loss function from the AlphaZero paper.
    It ONLY works in discrete settings.

    Attributes
    ----------
    policy_coeff: float
        Scaling factor for the policy component.
    value_coeff: float
        Scaling factor for the value component.
    reduction: str
        Defines how the loss is reduced to a scalar. Can be either "sum" or "mean".
    """

    policy_coeff: float
    value_coeff: float
    reduction: str

    def __init__(self, policy_coeff: float, value_coeff: float, reduction: str) -> None:
        """Constructor.

        Parameters
        ----------
        policy_coeff: float
            Scaling factor for the policy component.
        value_coeff: float
            Scaling factor for the value component.
        reduction: str
            Defines how the loss is reduced to a scalar. Can be either "sum" or "mean".
        """
        super().__init__()

        self.name = type(self).__name__

        self.policy_coeff = policy_coeff
        self.value_coeff = value_coeff
        self.reduction = reduction

    def _calculate_policy_loss(  # type: ignore[override]
        self, pi_prior_logits: torch.Tensor, pi_mcts: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the policy loss.

        The policy loss is the cross-entropy between the network probability distribution
        and the action with the highest visitation counts.

        Parameters
        ----------
        pi_prior_logits: torch.Tensor
            Prior distribution over the available actions from the network.
        pi_mcts: torch.Tensor
            Improved MCTS policy for the same state.

        Returns
        -------
        torch.Tensor
            Policy loss as scalar tensor.
        """
        # first we have to convert the probabilities to labels
        pi_mcts = pi_mcts.argmax(dim=1)
        pi_loss = F.cross_entropy(pi_prior_logits, pi_mcts, reduction=self.reduction)
        return pi_loss

    def _calculate_value_loss(
        self, V_hat: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the value loss of AlphaZero.

        The value loss is the mean squared error between the value estimate of the
        neural network and an improved value target produced by the MCTS.

        Parameters
        ----------
        V_hat: torch.Tensor
            Value estimates from the neural network for the training state.
        V: torch.Tensor
            V
            Improved value targets for that state.

        Returns
        -------
        torch.Tensor
            Value loss as scalar tensor.
        """
        return F.mse_loss(V_hat, V, reduction=self.reduction)

    def forward(  # type: ignore[override]
        self,
        pi_prior_logits: torch.Tensor,
        pi_mcts: torch.Tensor,
        V_hat: torch.Tensor,
        V: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Short description

        Longer description

        Parameters
        ----------
        pi_prior_logits: torch.Tensor
            Prior probabilities for all actions from the neural network.
        pi_mcts: torch.Tensor
            Normalized MCTS visitation counts for the selected actions.
        V_hat: torch.Tensor
            Neural network value estimates.
        V: torch.Tensor
            Value targets.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary holding the scalar loss values as values and the name of the
            component as key.
        """
        policy_loss = self.policy_coeff * self._calculate_policy_loss(
            pi_prior_logits, pi_mcts
        )
        value_loss = self.value_coeff * self._calculate_value_loss(V_hat, V)
        loss = policy_loss + value_loss
        return {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss}


class A0CLoss(Loss):
    """Implementation of the A0C loss.

    A0C is an extension of AlphaZero for continuous action spaces. It formulates a continuous
    training target and adds an entropy loss component to prevent the distribution from
    collapsing. More information is in the paper: https://arxiv.org/pdf/1805.09613.pdf.

    Attributes
    ----------
    tau: float
        Temperature parameter for the log-visitation counts in the policy loss.
    policy_coeff: float
        Scaling factor for the policy component of the loss.
    alpha: Union[float, torch.Tensor]
        Scaling factor for the entropy regularization term.
    value_coeff: float
        Scaling factor for the value component of the loss.
    reduction: str
        Defines how the loss is reduced to a scalar. Can be either "sum" or "mean".
    """

    tau: float
    policy_coeff: float
    alpha: Union[float, torch.Tensor]
    value_coeff: float
    reduction: str

    def __init__(
        self,
        tau: float,
        policy_coeff: float,
        alpha: Union[float, torch.Tensor],
        value_coeff: float,
        reduction: str,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        tau: float
            Temperature parameter for the log-visitation counts in the policy loss.
        policy_coeff: float
            Scaling factor for the policy component of the loss.
        alpha: Union[float, torch.Tensor]
            Scaling factor for the entropy regularization term.
        value_coeff: float
            Scaling factor for the value component of the loss.
        reduction: str
            Defines how the loss is reduced to a scalar. Can be either "sum" or "mean".
        """
        super().__init__()
        self.tau = tau
        self.policy_coeff = policy_coeff
        self.alpha = alpha
        self.value_coeff = value_coeff
        self.reduction = reduction

    def _calculate_policy_loss(  # type: ignore[override]
        self, log_probs: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        """Implements the A0C policy loss.

        The A0C policy loss uses the REINFORCE trick to move the continuous network
        distribution closer to a distribution specified by the normalized visitation counts.
        More information is in the paper: https://arxiv.org/pdf/1805.09613.pdf.

        Parameters
        ----------
        log_probs: torch.Tensor
            Action log-probabilities from the network's policy distribution.
        counts: torch.Tensor
            Action visitation counts

        Returns
        -------
        torch.Tensor
            Reduced policy loss.
        """
        with torch.no_grad():
            # calculate scaling term
            try: 
                log_diff = log_probs - self.tau * torch.log(counts)
            except:
                import ipdb
                ipdb.set_trace()

        # multiply with log_probs gradient
        policy_loss = torch.einsum("ni, ni -> n", log_diff, log_probs)

        if self.reduction == "mean":
            return policy_loss.mean()
        else:
            return policy_loss.sum()

    def _calculate_value_loss(
        self, V_hat: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the value loss of A0C.

        The value loss is the same as in the original AlphaZero paper.

        Parameters
        ----------
        V_hat: torch.Tensor
            Value estimates from the neural network for the training state.
        V: torch.Tensor
            V
            Improved value targets for that state.

        Returns
        -------
        torch.Tensor
            Value loss as scalar tensor.
        """
        return F.mse_loss(V_hat, V, reduction=self.reduction)

    def _calculate_entropy_loss(self, entropy: torch.Tensor) -> torch.Tensor:
        """Calculate the entropy regularization term.

        The entropy of a distribution can be approximated through the action log-probabilities.
        Note: Analytical computation is not possible for the squashed normal distribution
        or a GMM policy.

        Parameters
        ----------
        entropy: torch.Tensor
            Entropy as output from the policy network.

        Returns
        -------
        torch.Tensor
            Entropy regularization term as scalar Tensor.
        """
        if self.reduction == "mean":
            return entropy.mean()
        else:
            return entropy.sum()

    def forward(  # type: ignore[override]
        self,
        log_probs: torch.Tensor,
        counts: torch.Tensor,
        entropy: torch.Tensor,
        V: torch.Tensor,
        V_hat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Calculate the A0C loss.

        Parameters
        ----------
        log_probs: torch.Tensor
            Action log probabilities from the network policy given a state.
        counts: torch.Tensor
            Action visitation counts.
        entropy: torch.Tensor
            Approximate Entropy of the neural network distribution for a given state.
        V_hat: torch.Tensor
            Neural network value estimates.
        V: torch.Tensor
            Value targets.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary holding the loss component name as keys and the loss value for the component
            as scalar Tensor.
        """
        policy_loss = self.policy_coeff * self._calculate_policy_loss(log_probs, counts)
        value_loss = self.value_coeff * self._calculate_value_loss(V_hat, V)
        entropy_loss = self.alpha * self._calculate_entropy_loss(entropy)
        loss = policy_loss + entropy_loss + value_loss
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
        }


class A0CLossTuned(A0CLoss):
    """Tuned version of the A0C loss using automatic entropy tuning from SAC.

    This class is the same as the A0C loss except that the temperature for the entropy
    regularization term is adjusted automatically over the course of the training.
    Since the temperature parameter can never be negative, log-alpha is learned and
    then exponentiated.
    More information is in the second SAC paper: https://arxiv.org/pdf/1812.05905.pdf.

    Attributes
    ----------
    tau: float
        Temperature parameter for the log-visitation counts in the policy loss.
    policy_coeff: float
        Scaling factor for the policy component of the loss.
    alpha: Union[float, torch.Tensor]
        Scaling factor for the entropy regularization term.
    value_coeff: float
        Scaling factor for the value component of the loss.
    reduction: str
        Defines how the loss is reduced to a scalar. Can be either "sum" or "mean".
    clip: float
        Gradient clipping value for the alpha loss.
    device: torch.device
        Device for the loss. Can be either "cpu" or "cuda".
    log_alpha: torch.Tensor
        Log parameter that is actually learned.
    optimizer: torch.optim.Optimizer
        Torch optimizer for adjusting the log-alpha parameter.
    """

    tau: float
    policy_coeff: float
    alpha: torch.Tensor
    value_coeff: float
    reduction: str
    clip: float
    device: torch.device
    log_alpha: torch.Tensor
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        action_dim: int,
        alpha_init: float,
        lr: float,
        tau: float,
        policy_coeff: float,
        value_coeff: float,
        reduction: str,
        grad_clip: float,
        device: str,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        action_dim: int
            Dimensionality of the action space. Used as target for tuning alpha.
        alpha_init: float
            Initial value for alpha.
        lr: float
            Alpha optimizer learning rate.
        tau: float
            Temperature parameter for the log-visitation counts in the policy loss.
        policy_coeff: float
            Scaling factor for the policy component of the loss.
        value_coeff: float
            Scaling factor for the value component of the loss.
        reduction: str
            Defines how the loss is reduced to a scalar. Can be either "sum" or "mean".
        grad_clip: float
            Gradient clipping value for the alpha loss.
        device: torch.device
            Device for the loss. Can be either "cpu" or "cuda".
        """
        self.clip = grad_clip
        self.device = torch.device(device)

        # set target entropy to -|A|
        self.target_entropy = -action_dim
        # initialize alpha to 1
        self.log_alpha = torch.tensor(
            np.log(alpha_init),
            requires_grad=True,
            device=self.device,
            dtype=torch.float32,
        )

        self.alpha = self.log_alpha.exp()

        self.optimizer = Adam([self.log_alpha], lr=lr)

        # for simplicity: Use the same optimizer settings as for the neural network
        super().__init__(
            tau=tau,
            policy_coeff=policy_coeff,
            alpha=self.alpha,
            value_coeff=value_coeff,
            reduction=reduction,
        )

    def _update_alpha(self, entropy: torch.Tensor) -> torch.Tensor:
        """Perform an update state for the entropy regularization term temperature parameter
        alpha.

        Parameters
        ----------
        entropy: torch.Tensor
            Approximate policy distribution entropy from the network.

        Returns
        -------
        torch.Tensor
            Alpha loss as scalar Tensor.
        """
        self.log_alpha.grad = None
        # calculate loss for entropy regularization parameter
        alpha_loss = (self.alpha * (entropy - self.target_entropy).detach()).mean()
        alpha_loss.backward()

        if self.clip:
            clip_grad_norm(self.log_alpha, self.clip)
        self.optimizer.step()

        self.alpha = self.log_alpha.exp()

        return alpha_loss.detach().cpu()

    def forward(  # type: ignore[override]
        self,
        log_probs: torch.Tensor,
        counts: torch.Tensor,
        entropy: torch.Tensor,
        V: torch.Tensor,
        V_hat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Calculate the A0C loss and perform an alpha update step step.

        Parameters
        ----------
        log_probs: torch.Tensor
            Action log probabilities from the network policy given a state.
        counts: torch.Tensor
            Action visitation counts.
        entropy: torch.Tensor
            Approximate Entropy of the neural network distribution for a given state.
        V_hat: torch.Tensor
            Neural network value estimates.
        V: torch.Tensor
            Value targets.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary holding the loss component name as keys and the loss value for the component
            as scalar Tensor.
        """
        policy_loss = self.policy_coeff * self._calculate_policy_loss(log_probs, counts)
        value_loss = self.value_coeff * self._calculate_value_loss(V_hat, V)
        entropy_loss = self.alpha.detach().item() * self._calculate_entropy_loss(
            entropy
        )
        loss = policy_loss + entropy_loss + value_loss
        alpha_loss = self._update_alpha(entropy)
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            "alpha_loss": alpha_loss,
        }


class RPOLoss(A0CLossTuned):
    pass

class RPOAdvantageLoss(A0CLossTuned):
    # def KL_Divergence(self, mu_1, sigma_1, mu_2, sigma_2):
    #     pass
    def forward(  # type: ignore[override]
        self, tensor_obs: Dict, train_data_dict: Dict):

        # policy_loss = self.policy_coeff * self._calculate_policy_loss(log_probs, counts)
        policy_loss = self.policy_coeff * self._calculate_policy_loss(tensor_obs, train_data_dict)
        value_loss = self.value_coeff * self._calculate_value_loss(
            train_data_dict['V_hat'], tensor_obs['values'])
        entropy_loss = self.alpha.detach().item() * self._calculate_entropy_loss(
            train_data_dict['entropy']
        )
        loss = policy_loss + entropy_loss + value_loss
        # loss = policy_loss + value_loss
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            # "alpha_loss": alpha_loss,
        }
    def _calculate_policy_loss(  # type: ignore[override]
        self, tensor_obs, train_data_dict
    ) -> torch.Tensor:
        # batch_size = tensor_obs['values'].shape[0]
        total_advantage = -torch.stack([
            (a*p).sum() 
            for a, p in zip(tensor_obs['advantage'], train_data_dict['probs'])
        ])
        c_uct = 0.05
        kl = c_uct/(tensor_obs['n'])**0.5*train_data_dict['kl_divergence']
        assert kl.shape == total_advantage.shape
        policy_loss = total_advantage + kl
        # policy_loss = (tensor_obs['n'])**0.5*total_advantage + train_data_dict['kl_divergence']


        if self.reduction == "mean":
            return policy_loss.mean()
        else:
            return policy_loss.sum()

    # def _calculate_policy_loss(  # type: ignore[override]
    #     self, tensor_obs, train_data_dict
    # ) -> torch.Tensor:
    #     # batch_size = tensor_obs['values'].shape[0]
    #     with torch.no_grad():
    #         log_diff = [
    #             # (torch.log(p+epsilon) - self.tau*torch.log(count + 1))#.sum() 
    #             (torch.log(p) - self.tau*torch.log(count))#.sum() 
    #             for count, p in zip(tensor_obs['counts'], train_data_dict['probs'])
    #         ]
    #     total_advantage = torch.stack([
    #         # (a*p).sum() 
    #         (a*torch.log(p)).sum() 
    #         for a, p in zip(log_diff, train_data_dict['probs'])
    #     ])
    #     c_uct = 0.05
    #     kl = c_uct/(tensor_obs['n'])**0.5*train_data_dict['kl_divergence']
    #     assert kl.shape == total_advantage.shape
    #     policy_loss = total_advantage + kl


    #     if self.reduction == "mean":
    #         return policy_loss.mean()
    #     else:
    #         return policy_loss.sum()

class VolumeLoss_2(RPOAdvantageLoss):
    pass
    def _calculate_policy_loss(  # type: ignore[override]
        self, tensor_obs, train_data_dict
    ) -> torch.Tensor:
        # settings_list = ["alphazero", "advantage", "kl_forward", "kl_backward", "jensen"]
        # assert self.setting in settings_list

        c_uct = 1
        kl = c_uct/(tensor_obs['n'])**0.5*train_data_dict['kl_divergence']

        if self.setting == "alphazero":
            epsilon = 0.01
            with torch.no_grad():
                log_diff = [
                    # (torch.log(p+epsilon) - self.tau*torch.log(count + 1))#.sum() 
                    (torch.log(p) - self.tau*torch.log(count))#.sum() 
                    for count, p in zip(tensor_obs['counts'], train_data_dict['probs'])
                ]
            policy_loss = torch.stack([
                (a*torch.log(p)).sum()  
                # (a*torch.log(p) - p).sum()   #Plugging in formula for 
                for a, p in zip(log_diff, train_data_dict['probs'])
            ])
        elif self.setting == "advantage":
            policy_loss = -torch.stack([
                (a*p).sum() 
                for a, p in zip(tensor_obs['advantage'], train_data_dict['probs'])
            ])
            policy_loss = policy_loss + kl
        elif self.setting == "kl_backward":
            policy_loss = torch.stack([
                # (torch.log(a)*p).sum() 
                (torch.log(a)*p + p).sum() 
                # (-torch.log(a)*p - p).sum() 
                for a, p in zip(tensor_obs['counts'], train_data_dict['probs'])
            ])
        elif self.setting == "kl_backward_corrected":
            policy_loss = torch.stack([
                # (torch.log(a)*p).sum() 
                # (torch.log(a)*p + p).sum() 
                (-p*torch.log(a) + p*torch.log(p) - p).sum() 
                for a, p in zip(tensor_obs['counts'], train_data_dict['probs'])
            ])
        elif self.setting == "kl_forward":
            for p in train_data_dict['probs']:
                if not ((p==p).all() and (p > 0).all()):
                    ipdb.set_trace()
            for p in tensor_obs['counts']:
                if not ((p==p).all() and (p > 0).all()):
                    ipdb.set_trace()

            #Forward KL divergence equation
            total_advantage = torch.stack([
                (policy/policy.sum()*torch.log(p+0.001)).sum()  
                # (a*torch.log(p) - p).sum()   #Plugging in formula for 
                for policy, p in zip(tensor_obs['counts'], train_data_dict['probs'])
            ])
        elif self.setting == "jensen":
            policy_loss = torch.stack([
                # (torch.log(a)*p).sum() 
                (torch.log(a)*p + a*torch.log(p+0.000001)).sum() 
                # (torch.log(a)*p + p).sum() 
                for a, p in zip(tensor_obs['counts'], train_data_dict['probs'])
            ])
        else:
            print(f"Loss function {self.setting} not found")
            assert False
        
        # assert kl.shape == total_advantage.shape
        # policy_loss = total_advantage #+ kl
        # policy_loss = total_advantage 
        # policy_loss = policy_loss + self._calculate_entropy_loss(train_data_dict['entropy'])
        # policy_loss = policy_loss + kl


        if self.reduction == "mean":
            return policy_loss.mean()
        else:
            return policy_loss.sum()

    def _calculate_density_loss(self, tensor_obs, train_data_dict):
        c1 = 1
        c2 = 1-c1
        entropy_bonus = 0#self.alpha.detach().item()
        density_loss = train_data_dict['density']*(
            (1+entropy_bonus)*torch.log(train_data_dict['density']) 
            - torch.log(tensor_obs['prob']/tensor_obs['volume']) 
            - (1+entropy_bonus))

        # density_loss = (train_data_dict['density'])**2 + (1-train_data_dict['node_density'])**2
        # density_loss = (train_data_dict['density'])**2 + (0.0001*tensor_obs['prob']/tensor_obs['volume']-train_data_dict['node_density'])**2
        density_loss = (train_data_dict['density'])**2 - train_data_dict['node_density']
        density_loss = (train_data_dict['density'])**2 + (0.00001/tensor_obs['volume']-train_data_dict['node_density'])**2

        # density_loss = - train_data_dict['node_density']#- train_data_dict['node_density'] #+ 0.1*train_data_dict['node_density']**2
        # density_loss = torch.log(train_data_dict['density'])**2 - torch.log(train_data_dict['node_density'])
        # density_loss = (train_data_dict['density'])**2 + (0.00001*tensor_obs['prob']/tensor_obs['volume']-train_data_dict['node_density'])**2
        # density_loss += 

        # density_loss += 0.01*tensor_obs['volume']*(tensor_obs['base_prob'].mean()
        #     - train_data_dict['density'])**2 
        # density_loss += 0.01*self.alpha.detach().item() * ( #tensor_obs['volume']* 
        #     (tensor_obs['base_prob'].mean()
        #     - train_data_dict['density'])**2 )
        # density_loss += (train_data_dict['inv_density'])

        # density_loss = torch.log(train_data_dict['density'])
        # density_loss = density_loss - torch.log(train_data_dict['node_density'])

        if self.reduction == "mean":
            return density_loss.mean()
        else:
            return density_loss.sum()

    def forward(  # type: ignore[override]
        self, tensor_obs: Dict, train_data_dict: Dict):
        self.setting = "kl_backward"
        self.setting = "alphazero"
        # self.setting = "jensen"

        # policy_loss = self.policy_coeff * self._calculate_policy_loss(log_probs, counts)
        policy_loss = self.policy_coeff * self._calculate_policy_loss(tensor_obs, train_data_dict)
        value_loss = self.value_coeff * self._calculate_value_loss(
            train_data_dict['V_hat'], tensor_obs['values'])
        # entropy_loss = (self.policy_coeff+self.alpha.detach().item()) * self._calculate_entropy_loss(
        #     train_data_dict['entropy']
        # )
        density_loss = self._calculate_density_loss(tensor_obs, train_data_dict)

        entropy = self._calculate_entropy_loss(train_data_dict['entropy'])
        if self.setting == "advantage":
            entropy_coeff = 0
        elif self.setting == "alphazero":
            entropy_coeff = (self.alpha.detach().item()) 
        elif self.setting == "kl_backward":
            entropy_coeff = (self.policy_coeff) 
        elif self.setting == "kl_backward_corrected":
            # entropy_loss = (self.alpha.detach().item() - self.policy_coeff) * entropy
            # entropy_loss = (-self.policy_coeff) * entropy
            entropy_coeff = self.alpha.detach().item()
        elif self.setting == "kl_forward":
            entropy_coeff = 0
        elif self.setting == "jensen":
            entropy_coeff = (self.policy_coeff)

        entropy_loss = entropy_coeff*entropy

        loss = policy_loss + value_loss + entropy_loss + density_loss
        if not (loss == loss).all():
            import ipdb
            ipdb.set_trace()
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            # "alpha_loss": alpha_loss,
        }


class VolumeLoss_3(RPOAdvantageLoss):
    pass
# class VolumeLoss(Loss):
class VolumeLoss(A0CLossTuned):
    pass

class ExplorerLoss(VolumeLoss_2):
    # def _calculate_policy_loss(  # type: ignore[override]
    #     self, tensor_obs, train_data_dict
    # ) -> torch.Tensor:
    #     # settings_list = ["alphazero", "advantage", "kl_forward", "kl_backward", "jensen"]
    #     # assert self.setting in settings_list

    #     c_uct = 1
    #     epsilon = 0.01
    #     with torch.no_grad():
    #         log_diff = [
    #             # (torch.log(p+epsilon) - self.tau*torch.log(count + 1))#.sum() 
    #             (torch.log(p) - self.tau*torch.log(count))#.sum() 
    #             for count, p in zip(tensor_obs['counts'], train_data_dict['probs'])
    #         ]
    #     policy_loss = torch.stack([
    #         (a*torch.log(p)).sum()  
    #         # (a*torch.log(p) - p).sum()   #Plugging in formula for 
    #         for a, p in zip(log_diff, train_data_dict['probs'])
    #     ])

    #     if self.reduction == "mean":
    #         return policy_loss.mean()
    #     else:
    #         return policy_loss.sum()
    # pass
    def _calculate_policy_loss(  # type: ignore[override]
        self, tensor_obs, train_data_dict
    ) -> torch.Tensor:
        # batch_size = tensor_obs['values'].shape[0]
        with torch.no_grad():
            log_diff = [
                # (torch.log(p+epsilon) - self.tau*torch.log(count + 1))#.sum() 
                (torch.log(p) - self.tau*torch.log(count))#.sum() 
                for count, p in zip(tensor_obs['counts'], train_data_dict['probs'])
            ]
        total_advantage = torch.stack([
            # (a*p).sum() 
            (a*torch.log(p)).sum() 
            for a, p in zip(log_diff, train_data_dict['probs'])
        ])
        # 
        # total_advantage = -torch.stack([
        #     (a*p).sum() 
        #     for a, p in zip(tensor_obs['advantage'], train_data_dict['probs'])
        # ])
        c_uct = 0.05
        kl = c_uct/(tensor_obs['n'])**0.5*train_data_dict['kl_divergence']
        assert kl.shape == total_advantage.shape
        policy_loss = total_advantage + kl
        # policy_loss = (tensor_obs['n'])**0.5*total_advantage + train_data_dict['kl_divergence']


        if self.reduction == "mean":
            return policy_loss.mean()
        else:
            return policy_loss.sum()


    def forward(  # type: ignore[override]
        self, tensor_obs: Dict, train_data_dict: Dict):
        self.setting = "kl_backward"
        self.setting = "alphazero"
        policy_loss = self.policy_coeff * self._calculate_policy_loss(tensor_obs, train_data_dict)
        value_loss = self.value_coeff * self._calculate_value_loss(
            train_data_dict['V_hat'], tensor_obs['values'])
        # entropy_loss = (self.policy_coeff+self.alpha.detach().item()) * self._calculate_entropy_loss(
        #     train_data_dict['entropy']
        # )
        density_loss = self._calculate_density_loss(tensor_obs, train_data_dict)

        entropy = self._calculate_entropy_loss(train_data_dict['entropy'])
        entropy_coeff = (self.alpha.detach().item())#*1000 

        entropy_loss = entropy_coeff*entropy

        loss = policy_loss + value_loss + entropy_loss + density_loss
        if not (loss == loss).all():
            import ipdb
            ipdb.set_trace()
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            # "alpha_loss": alpha_loss,
        }


class FrequencyLoss(VolumeLoss_2):    
    # pass
    def _calculate_density_loss(self, tensor_obs, train_data_dict):
        density_loss = (train_data_dict['density'])**2 + (1+tensor_obs['epoch'].unsqueeze(1)-train_data_dict['node_density'])**2
        # density_loss = (1+tensor_obs['epoch'].unsqueeze(1)-train_data_dict['node_density'])**2
        # density_loss = (train_data_dict['density'])**2 + (1-train_data_dict['node_density'])**2
        if self.reduction == "mean":
            return density_loss.mean()
        else:
            return density_loss.sum()

class OneShotLoss(VolumeLoss_2):  
    # def _calculate_density_loss(self, tensor_obs, train_data_dict):
    #     return super().density_loss(tensor_obs, train_data_dict)*0
    # def _calculate_density_loss(self, tensor_obs, train_data_dict):
    #     density_loss = (
    #         (train_data_dict['density'])**2 
    #         + 1/tensor_obs['volume']*(tensor_obs['volume']-train_data_dict['node_density'])**2
    #     )
    #     return density_loss.mean() #+ (train_data_dict['node_density'].mean() - 1)**2

    def _calculate_density_loss(self, tensor_obs, train_data_dict):
        # return 0*(
        #     train_data_dict['density'].mean() +
        #     train_data_dict['node_density'].mean()
        # )
        density_loss = (
            # 1/tensor_obs['volume'].mean()*(tensor_obs['volume']-train_data_dict['node_density'])**2
            tensor_obs['local_volume'].mean()*(tensor_obs['volume']-train_data_dict['node_density'])**2
            # - tensor_obs['local_volume']*torch.log(train_data_dict['node_density'])/tensor_obs['epoch']**0.5
        )
        c=0.1
        # return (
        #     density_loss.mean()*0 
        #     + (train_data_dict['density'].mean() - 1)**2 
        #     - 0.1*((tensor_obs['epoch']+1)**(-0.5)*(
        #             torch.log(train_data_dict['density']) 
        #             - torch.log(train_data_dict['density'].mean())
        #             + torch.log(train_data_dict['node_density']) 
        #             - torch.log(train_data_dict['node_density'].mean())
        #         )).mean()

        #     # + c*(train_data_dict['node_density'].mean() - 1)**2            
        # ) 
        c = 100 
        c_uct = 25
        lam = c_uct/(tensor_obs['epoch']+1)**0.5
        new_density_loss = -(
            (tensor_obs['traj_value'] - tensor_obs['traj_value'].mean() - lam)*train_data_dict['node_density'] + 
            # (tensor_obs['traj_value'] - tensor_obs['traj_value'].mean())*train_data_dict['node_density'] + 
            lam*torch.log(train_data_dict['node_density']+0.000001)
        ).mean()
        # return ((tensor_obs['values'] - train_data_dict['node_density'])**2).mean()
        return (
            new_density_loss        
            + (train_data_dict['density'].mean() - 1)**2 
            + (train_data_dict['node_density'].mean() - 1)**2 
        )



        inv_vol_mean = 1/(tensor_obs['volume']).mean()
        # return (
        #     (-tensor_obs['volume']*tensor_obs['local_volume']*inv_vol_mean*torch.log(train_data_dict['node_density'])
        #      + train_data_dict['node_density'])
            
        # ).mean()

        return (
            density_loss.mean()
            + (train_data_dict['density'].mean() - 1)**2 
            + (train_data_dict['node_density'].mean() - 1)**2 
            # - ((tensor_obs['epoch']+1)**(-0.5)*(
            - c*((
                0
                    # +torch.log(train_data_dict['density']) 
                    # - torch.log(train_data_dict['density'].mean())
                    # + tensor_obs['volume']/tensor_obs['volume'].mean()*
                    + tensor_obs['volume']*tensor_obs['local_volume']*torch.log(train_data_dict['node_density']) 
                    # - torch.log(train_data_dict['node_density'].mean()) 
                    
                    # - train_data_dict['node_density']
                    
                )).mean()/(tensor_obs['volume']*tensor_obs['local_volume']).mean()
            # + c*train_data_dict['node_density'].mean()

            # + c*(train_data_dict['node_density'].mean() - 1)**2            
        ) 

        # # Reverse KL divergence loss
        # vol = (tensor_obs['local_volume']*tensor_obs['volume'])
        # vol = (tensor_obs['volume'])
        # normed_vol = vol/vol.mean()
        # return (( 
        #         -1*tensor_obs['local_volume']*normed_vol*torch.log(train_data_dict['node_density']) 
        #         +  tensor_obs['local_volume']*train_data_dict['node_density']
        #     ).mean() 
        #     # + (train_data_dict['density'].mean() - 1)**2 
        #     # + (train_data_dict['node_density'].mean() - 1)**2 
        # )
            

        # Forward KL divergence loss
        # vol = (tensor_obs['local_volume']*tensor_obs['volume'])
        # vol = (tensor_obs['volume'])
        # normed_vol = vol/vol.mean()
        # return (( 
        #         train_data_dict['node_density']*torch.log(train_data_dict['node_density']/normed_vol) 
        #         -  train_data_dict['node_density']
        #     ).mean() 
        #     + (0.1*(train_data_dict['density'] - 1)**2).mean()
        #     + (train_data_dict['density'].mean() - 1)**2 
        #     + (train_data_dict['node_density'].mean() - 1)**2 
        # )
            

    def _calculate_policy_loss(  # type: ignore[override]
        self, tensor_obs, train_data_dict
    ) -> torch.Tensor:
        # batch_size = tensor_obs['values'].shape[0]
        total_advantage = -torch.stack([
            # (a*p).sum() 
            (a/(1-tensor_obs['gamma'])*p).sum() 
            for a, p in zip(tensor_obs['advantage'], train_data_dict['probs'])
        ])
        c_uct = 10#0.05
        lam = c_uct/(tensor_obs['epoch']+1)**0.5
        kl = lam*train_data_dict['base_kl_divergence']
        square_div = lam*train_data_dict['square_divergence']
        base_js = lam*train_data_dict['base_js_divergence']
        js = c_uct*(tensor_obs['epoch']+1)**0.5*train_data_dict['js_divergence']
        # kl = 0.05*train_data_dict['base_kl_divergence']
        assert kl.shape == total_advantage.shape
        density_advantage = lam*torch.stack([
            (v*torch.log(p)).sum() 
            for v, p in zip(tensor_obs['children_unweighted_density'], train_data_dict['probs'])
        ])
        # policy_loss = total_advantage*0 + kl
        # policy_loss = total_advantage*0 + square_div
        policy_loss = total_advantage + js + base_js
        policy_loss = total_advantage + base_js
        # policy_loss = total_advantage*tensor_obs['base_prob'] + density_advantage + js + base_js
        # policy_loss = kl
        # policy_loss = total_advantage*0 + train_data_dict['kl_divergence']


        if self.reduction == "mean":
            return policy_loss.mean()
        else:
            return policy_loss.sum()



    def _calculate_value_loss(
        self, V_hat: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        return ((V_hat - V)**2).mean()
        return F.mse_loss(V_hat, V, reduction=self.reduction)

    def _calculate_value_reg(self, tensor_obs: Dict, V: torch.Tensor) -> torch.Tensor:
        return ((tensor_obs['epoch'] + 1)**(-0.5)*V**2).mean()

    def forward(  # type: ignore[override]
        self, tensor_obs: Dict, train_data_dict: Dict):

        policy_loss = self.policy_coeff * self._calculate_policy_loss(tensor_obs, train_data_dict)
        value_loss = self.value_coeff * self._calculate_value_loss(
            train_data_dict['V_hat'], tensor_obs['values'])
        value_reg = self.value_coeff * self._calculate_value_reg(tensor_obs, train_data_dict['V_hat'])
        density_loss = self._calculate_density_loss(tensor_obs, train_data_dict)

        entropy = self._calculate_entropy_loss(train_data_dict['entropy'])
        entropy_coeff = (self.alpha.detach().item()) 
        entropy_loss = entropy_coeff*entropy

        # import ipdb
        # ipdb.set_trace()
        # value_loss = ((train_data_dict['V_hat'] - 1*(tensor_obs['states'] > 0.5).all(dim=-1).unsqueeze(-1))**2).mean()
        loss = (policy_loss + value_loss + entropy_loss + density_loss)
        loss = (policy_loss + value_loss + 0*entropy_loss + density_loss)
        # loss = entropy_loss
        # loss = (policy_loss*0 + value_loss*0 + entropy_loss*0 + 0*density_loss)
        # loss = (policy_loss + value_loss + entropy_loss + 0*density_loss)
        # loss = (policy_loss + value_loss + value_reg + entropy_loss + density_loss)
        # loss = (0*policy_loss + value_loss + entropy_loss + density_loss)
        if not (loss == loss).all():
            import ipdb
            ipdb.set_trace()
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            # "alpha_loss": alpha_loss,
        }


class BetterBufferLoss(OneShotLoss):  
    def _calculate_density_loss(self, tensor_obs, train_data_dict):
        density_loss = (
            tensor_obs['local_volume'].mean()*(tensor_obs['volume']-train_data_dict['node_density'])**2
        )
        c = 100 
        c_uct = 25
        lam = c_uct/(tensor_obs['epoch']+1)**0.5
        new_density_loss = -(
            (tensor_obs['traj_value'] - tensor_obs['traj_value'].mean() - lam)*train_data_dict['node_density'] + 
            lam*torch.log(train_data_dict['node_density']+0.000001)
        ).mean()
        return (
            new_density_loss        
            + (train_data_dict['density'].mean() - 1)**2 
            + (train_data_dict['node_density'].mean() - 1)**2 
        )



        # inv_vol_mean = 1/(tensor_obs['volume']).mean()
        # return (
        #     density_loss.mean()*0 
        #     + (train_data_dict['density'].mean() - 1)**2 
        #     + (train_data_dict['node_density'].mean() - 1)**2 
        #     - c*((
        #         0 + tensor_obs['volume']*tensor_obs['local_volume']*torch.log(train_data_dict['node_density'])                 
        #         )).mean()/(tensor_obs['volume']*tensor_obs['local_volume']).mean()    
        # ) 

    def _calculate_policy_loss(  # type: ignore[override]
        self, tensor_obs, train_data_dict
    ) -> torch.Tensor:
        # total_advantage = -torch.stack([
        #     (a/(1-tensor_obs['gamma'])*p).sum() 
        #     for a, p in zip(tensor_obs['advantage'], train_data_dict['probs'])
        # ])
        total_advantage = tensor_obs['advantage']/(1-tensor_obs['gamma'])*train_data_dict['probs']
        total_advantage = tensor_obs['policy']*train_data_dict['log_probs']
        c_uct = 10#0.05
        lam = c_uct/(tensor_obs['epoch']+1)**0.5
        kl = lam*train_data_dict['base_kl_divergence']
        square_div = lam*train_data_dict['square_divergence']
        base_js = lam*train_data_dict['base_js_divergence']
        js = c_uct*(tensor_obs['epoch']+1)**0.5*train_data_dict['js_divergence']
        # kl = 0.05*train_data_dict['base_kl_divergence']
        try: 
            assert kl.shape == total_advantage.shape
        except: 
            import ipdb
            ipdb.set_trace()
        # density_advantage = lam*torch.stack([
        #     (v*torch.log(p)).sum() 
        #     for v, p in zip(tensor_obs['children_unweighted_density'], train_data_dict['probs'])
        # ])
        # policy_loss = total_advantage + js + base_js
        # policy_loss = total_advantage + base_js
        total_advantage = torch.exp(tensor_obs['advantage']/(lam*(1-tensor_obs['gamma'])))*train_data_dict['log_probs']
        policy_loss = total_advantage + kl

        if self.reduction == "mean":
            return policy_loss.mean()
        else:
            return policy_loss.sum()



    def _calculate_value_loss(
        self, V_hat: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        return ((V_hat - V)**2).mean()
        return F.mse_loss(V_hat, V, reduction=self.reduction)

    def _calculate_value_reg(self, tensor_obs: Dict, V: torch.Tensor) -> torch.Tensor:
        return ((tensor_obs['epoch'] + 1)**(-0.5)*V**2).mean()

    def forward(  # type: ignore[override]
        self, tensor_obs: Dict, train_data_dict: Dict):
        # import ipdb
        # ipdb.set_trace()

        policy_loss = self.policy_coeff * self._calculate_policy_loss(tensor_obs, train_data_dict)
        value_loss = self.value_coeff * self._calculate_value_loss(
            train_data_dict['V_hat'], tensor_obs['values'])
        value_reg = self.value_coeff * self._calculate_value_reg(tensor_obs, train_data_dict['V_hat'])
        # density_loss = self._calculate_density_loss(tensor_obs, train_data_dict)

        entropy = self._calculate_entropy_loss(train_data_dict['entropy'])
        entropy_coeff = (self.alpha.detach().item()) 
        entropy_loss = entropy_coeff*entropy

        # import ipdb
        # ipdb.set_trace()
        # value_loss = ((train_data_dict['V_hat'] - 1*(tensor_obs['states'] > 0.5).all(dim=-1).unsqueeze(-1))**2).mean()
        loss = (policy_loss + value_loss + entropy_loss)# + density_loss)
        loss = (policy_loss + value_loss + 0*entropy_loss)# + density_loss)
        # loss = entropy_loss
        # loss = (policy_loss*0 + value_loss*0 + entropy_loss*0 + 0*density_loss)
        # loss = (policy_loss + value_loss + entropy_loss + 0*density_loss)
        # loss = (policy_loss + value_loss + value_reg + entropy_loss + density_loss)
        # loss = (0*policy_loss + value_loss + entropy_loss + density_loss)
        if not (loss == loss).all():
            import ipdb
            ipdb.set_trace()
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            # "alpha_loss": alpha_loss,
        }


class HERLoss(OneShotLoss):  
    def forward(  # type: ignore[override]
        self, tensor_obs: Dict, train_data_dict: Dict):

        policy_loss = self.policy_coeff * self._calculate_policy_loss(tensor_obs, train_data_dict)
        value_loss = self.value_coeff * self._calculate_value_loss(
            train_data_dict['V_hat'], tensor_obs['values'])

        HER_value_loss = 0.1 * self.value_coeff * self._calculate_value_loss(
            train_data_dict['HER_V_hat'], tensor_obs['HER_targets'])

        value_reg = self.value_coeff * self._calculate_value_reg(tensor_obs, train_data_dict['V_hat'])
        density_loss = self._calculate_density_loss(tensor_obs, train_data_dict)

        loss = (policy_loss + value_loss + HER_value_loss + density_loss)
        if not (loss == loss).all():
            import ipdb
            ipdb.set_trace()
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            # "alpha_loss": alpha_loss,
        }