from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

def generate(model: nn.Module,
             input_ids: torch.Tensor,
             max_length: int,
             num_beams: int = 1,
             do_sample: bool = True,
             early_stopping: bool = False,
             eos_token_id: Optional[int] = None,
             pad_token_id: Optional[int] = None,
             top_k: Optional[int] = None,
             top_p: Optional[float] = None,
             temperature: Optional[float] = None,
             prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
             update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
             **model_kwargs) -> torch.Tensor:
    
    def sample(model: nn.Module,
            input_ids: torch.Tensor,
            max_length: int,
            early_stopping: bool = False,
            eos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            temperature: Optional[float] = None,
            prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
            update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
            **model_kwargs) -> torch.Tensor:
        if input_ids.size(1) >= max_length:
            return input_ids

        logits_processor = prepare_logits_processor(top_k, top_p, temperature) #
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        for _ in range(input_ids.size(1), max_length):
            model_inputs = prepare_inputs_fn(input_ids, **model_kwargs) if prepare_inputs_fn is not None else {
                'input_ids': input_ids
            }
            outputs = model(**model_inputs)

            next_token_logits = outputs['logits'][:, -1, :]
            # pre-process distribution
            next_token_logits = logits_processor(input_ids, next_token_logits)
            # sample
            probs = torch.softmax(next_token_logits, dim=-1, dtype=torch.float)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if update_model_kwargs_fn is not None:
                model_kwargs = update_model_kwargs_fn(outputs, **model_kwargs)

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished if early_stopping=True
            if early_stopping and _is_sequence_finished(unfinished_sequences):
                break

        return input_ids
    
    return sample(model,
                      input_ids,
                      max_length,
                      early_stopping=early_stopping,
                      eos_token_id=eos_token_id,
                      pad_token_id=pad_token_id,
                      top_k=top_k,
                      top_p=top_p,
                      temperature=temperature,
                      prepare_inputs_fn=prepare_inputs_fn,
                      update_model_kwargs_fn=update_model_kwargs_fn,
                      **model_kwargs)

class Actor(nn.Module):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self, model: nn.Module, lora_rank: int = 0, lora_train_bias: str = 'none') -> None:
        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.convert_to_lora()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        return_action_mask: bool = True,
        **kwargs
    ) -> Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:
        sequences = generate(self.model, input_ids, **kwargs)
        attention_mask = None
        pad_token_id = kwargs.get('pad_token_id', None)
        if pad_token_id is not None:
            attention_mask = sequences.not_equal(pad_token_id).to(dtype=torch.long, device=sequences.device)
        if not return_action_mask:
            return sequences, attention_mask, None
        input_len = input_ids.size(1)
        eos_token_id = kwargs.get('eos_token_id', None)
        if eos_token_id is None:
            action_mask = torch.ones_like(sequences, dtype=torch.bool)
        else:
            # left padding may be applied, only mask action
            action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
            action_mask = F.pad(action_mask, (1 + input_len, -1), value=True)    # include eos token and input
        action_mask[:, :input_len] = False
        action_mask = action_mask[:, 1:]
        return sequences, attention_mask, action_mask[:, -(sequences.size(1) - input_len):]

    def forward(self,
                sequences: torch.LongTensor,
                num_actions: int,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns action log probs
        """
        output = self.model(sequences, attention_mask=attention_mask)
        logits = output['logits']
        log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
        return log_probs[:, -num_actions:]

    def get_base_model(self):
        return self.model  # model = GPT2LMHeadModel.from_pretrained(pretrained)


def compute_approx_kl(log_probs: torch.Tensor,
                      log_probs_base: torch.Tensor,
                      action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.  计算两个分布之间的近似KL发散。
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    approx_kl = (log_ratio.exp() - 1) - log_ratio
    if action_mask is not None:
        approx_kl = masked_mean(approx_kl, action_mask, dim=1)
        return approx_kl
    approx_kl = approx_kl.mean(dim=1)
    return approx_kl

def compute_reward(r: Union[torch.Tensor, float],
                   kl_coef: float, 
                   log_probs: torch.Tensor,
                   log_probs_base: torch.Tensor,
                   action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if kl_coef <= 0.0:
        return r
    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    reward = r - kl_coef * kl
    return reward

class NaiveExperienceMaker(ExperienceMaker):
    """
    Naive experience maker.
    """

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        sequences, attention_mask, action_mask = self.actor.generate(input_ids,
                                                                     return_action_mask=True,
                                                                     **generate_kwargs)
        num_actions = action_mask.size(1)

        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        
        value = self.critic(sequences, action_mask, attention_mask)
        
        r = self.reward_model(sequences, attention_mask)
        reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask)

        advantage = reward - value
        # TODO(ver217): maybe normalize adv
        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)

        return Experience(sequences, action_log_probs, value, reward, advantage, attention_mask, action_mask)
