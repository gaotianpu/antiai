# from: https://github.com/hpcaitech/ColossalAI/blob/main/applications/Chat/coati/models/loss.py
# 参考：https://spinningup.openai.com/en/latest/algorithms/ppo.html
from typing import Optional

import torch
import torch.nn as nn

# from .utils import masked_mean
def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    return mean

class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss, 单个生成token的概率
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous() #batch,seq,vocab_prob ?
        shift_labels = labels[..., 1:].contiguous() #batch,token 
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(self,
                log_probs: torch.Tensor,
                old_log_probs: torch.Tensor,
                advantages: torch.Tensor, #优势估计
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        ratio = (log_probs - old_log_probs).exp() #当前策略值 - 旧策略值
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        if action_mask is not None:
            loss = masked_mean(loss, action_mask)
        loss = loss.mean()
        return loss


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.4) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(self,
                values: torch.Tensor,
                old_values: torch.Tensor,
                reward: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
        surr1 = (values_clipped - reward)**2
        surr2 = (values - reward)**2
        loss = torch.max(surr1, surr2)
        loss = loss.mean()
        return 0.5 * loss


class PPOPtxActorLoss(nn.Module):
    """
    To Do:

    PPO-ptx Actor Loss
    """

    def __init__(self, policy_clip_eps: float = 0.2, pretrain_coef: float = 0.0, pretrain_loss_fn=GPTLMLoss()) -> None:
        super().__init__()
        self.pretrain_coef = pretrain_coef
        self.policy_loss_fn = PolicyLoss(clip_eps=policy_clip_eps)
        self.pretrain_loss_fn = pretrain_loss_fn

    def forward(self,
                log_probs: torch.Tensor,
                old_log_probs: torch.Tensor,
                advantages: torch.Tensor,
                lm_logits: torch.Tensor,
                lm_input_ids: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        policy_loss = self.policy_loss_fn(log_probs, old_log_probs, advantages, action_mask=action_mask)
        lm_loss = self.pretrain_loss_fn(lm_logits, lm_input_ids)
        return policy_loss + self.pretrain_coef * lm_loss


class LogSigLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2203.02155
    """

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs)
        loss = -log_probs.mean()
        return loss


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss
