import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def transpose_batch_time(inputs):
    r"""Transposes inputs between time-major and batch-major.
    """
    return inputs.transpose(0, 1)


def mask_sequences(
    sequence,
    sequence_length,
    dtype=None,
    time_major=False
):
    if not torch.is_tensor(sequence):
        sequence = torch.tensor(sequence, dtype=dtype)
    sequence: torch.Tensor

    rank = sequence.dim()
    if rank < 2:
        raise ValueError("`sequence` must be 2D or higher order.")

    if time_major:
        sequence = transpose_batch_time(sequence)
    max_time = sequence.size(1)
    if dtype is None:
        dtype = sequence.dtype
    mask = sequence_mask(sequence_length, max_time, dtype=dtype)
    mask = mask.view(*mask.size(), *([1] * (rank - 2)))
    sequence = sequence * mask
    if time_major:
        sequence = transpose_batch_time(sequence)

    return sequence


def reduce_batch_time(
    sequence,
    sequence_length,
    average_across_batch=True,
    average_across_timesteps=False,
    sum_over_batch=False,
    sum_over_timesteps=True
):
    if average_across_timesteps and sum_over_timesteps:
        raise ValueError("Only one of `average_across_timesteps` and "
                         "`sum_over_timesteps` can be set.")
    if average_across_batch and sum_over_batch:
        raise ValueError("Only one of `average_across_batch` and "
                         "`sum_over_batch` can be set.")

    if sum_over_timesteps:
        sequence = torch.sum(sequence, dim=1)
    elif average_across_timesteps:
        if sequence_length is None:
            sequence = torch.mean(sequence, dim=1)
        else:
            sequence = (torch.sum(sequence, dim=1).float() /
                        sequence_length.float())

    if sum_over_batch:
        sequence = torch.sum(sequence, dim=0)
    elif average_across_batch:
        sequence = torch.mean(sequence, dim=0)

    return sequence


def mask_and_reduce(
    sequence,
    sequence_length,
    average_across_batch=True,
    average_across_timesteps=False,
    sum_over_batch=False,
    sum_over_timesteps=True,
):
    sequence = mask_sequences(sequence,
                              sequence_length,
                              dtype=None,
                              time_major=False)

    return reduce_batch_time(sequence,
                             sequence_length,
                             average_across_batch,
                             average_across_timesteps,
                             sum_over_batch,
                             sum_over_timesteps)


def label_smoothed_nll_loss(lprobs, target, probs, epsilon, ignore_index=None, reduce=True):
    """
    Args:
        lprobs (Tensor): [batch_size * max_tgt_len, vocab_size]
        target (Tensor): [batch_size * max_tgt_len]
        probs (Tensor): [batch_size * max_tgt_len, vocab_size]
        
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1) # [batch_size * max_tgt_len, 1]
    nll_loss = -lprobs.gather(dim=-1, index=target) # [batch_size * max_tgt_len, 1]
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True) # [batch_size * max_tgt_len, 1]

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss


def sequence_mask(lengths, max_len=None, dtype=None, device=None) :
    r"""Return a mask tensor representing the first N positions of each cell.
    If ``lengths`` has shape ``[d_1, d_2, ..., d_n]`` the resulting tensor
    ``mask`` has dtype ``dtype`` and shape ``[d_1, d_2, ..., d_n, maxlen]``,
    with
    ```
    mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
    ```
    Examples:
    ```python
    sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                 #  [True,  True,  True, False, False],
                                 #  [True,  True, False, False, False]]
    sequence_mask([[1, 3],[2,0]])  # [[[ True, False, False],
                                   #   [ True,  True,  True]],
                                   #  [[ True,  True, False],
                                   #   [False, False, False]]]
    ```
    Args:
        lengths: integer tensor or list of int, all its values <= max_len.
        max_len: scalar integer tensor, size of last dimension of returned
            tensor. Default is the maximum value in ``lengths``.
        dtype: the desired data type of returned tensor. Default: if None,
            returns :torch:`ByteTensor`.
        device: the desired device of returned tensor. Default: if None, uses
            the current device for the default tensor type.
    Returns:
        A mask tensor of shape :python:`lengths.shape + (max_len,)`, cast to
        specified dtype.
    Raises:
        ValueError: if ``max_len`` is not a scalar.
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, device=device)
    elif device is None:
        device = lengths.device
    lengths: torch.LongTensor
    if max_len is None:
        max_len = torch.max(lengths).item()

    size = lengths.size()
    row_vector = torch.arange(max_len, device=device, dtype=lengths.dtype).view(
        *([1] * len(size)), -1).expand(*size, max_len)
    mask = (row_vector < lengths.unsqueeze(-1)).to(device=device)
    if dtype is not None:
        mask = mask.to(dtype=dtype)

    return mask


def masked_reverse_cumsum(X, lengths, dim):
    """
    Args:
        X (Tensor): [batch_size, max_tgt_len]
        lengths (Tensor): [batch_size]
        dim (int): -1
        gamma (float): the discount factor
    
    """
    masked_X = X * sequence_mask(lengths, max_len=X.shape[1])
    return (masked_X
            .flip(dims=[dim])
            .cumsum(dim=dim)
            .flip(dims=[dim]))


def discounted_future_sum(values, lengths, num_steps=None, gamma=1.0):
    """
    Args:
        values (Tensor): [batch_size, max_tgt_len]
        lengths (Tensor): [batch_size]
        num_steps (int): A positive integer number of future steps to sum
        gamma (float): A float discount value
    
    """
    assert values.dim() == 2
    
    batch_size, total_steps = values.shape
    values = values * sequence_mask(lengths, max_len=values.shape[1])

    num_steps = total_steps if num_steps is None else num_steps
    num_steps = min(num_steps, total_steps)
    
    padding = torch.zeros([batch_size, num_steps - 1]).to(values)
    padded_values = torch.cat([values, padding], 1)
    discount_filter = gamma ** torch.arange(num_steps).to(values).reshape(1, 1, -1)

    output = F.conv1d(padded_values.unsqueeze(-2), discount_filter).squeeze(1)
    return output


def get_reward_shaping_func(
    old_min: float,
    old_max: float,
    new_min: float,
    new_max: float
):
    def _shaping_func(rewards):
        percentile = (rewards - old_min) / (old_max - old_min)
        return percentile * (new_max - new_min) + new_min

    return _shaping_func


def single_step_PCL_loss(
    logits, 
    logits_, 
    actions, 
    rewards, 
    seq_lens, 
    gamma=1.0, 
    tau=1.0
):
    """
    Single-step unified path consistency learning (PCL). 
    
    See paper https://arxiv.org/pdf/2106.07704.pdf (Eq.15).

    Args:
        logits (Tensor): [batch_size, tgt_len, vocab_size]
        logits_ (Tensor): [batch_size, tgt_len, vocab_size]
        actions (Tensor): [batch_size, tgt_len]
        rewards (Tensor): [batch_size] or [batch_size, tgt_len]
        seq_lens (Tensor): [batch_size]
        gamma (float): the discount factor
        tau (float): Shannon entropy coefficient term

    """
    if rewards.dim() == 1:
        rewards_ = torch.zeros_like(actions).to(rewards)
        rewards_[torch.arange(seq_lens.shape[0]), 
                 seq_lens - 1] = rewards
        rewards = rewards_
    
    if logits.dim() == actions.dim() + 1:
        actions = actions.unsqueeze(-1)
    
    # calculate policy pi, which equals the advantage function
    Q = logits.gather(dim=-1, index=actions).squeeze(-1)
    V = tau * (logits / tau).logsumexp(dim=-1)
    A = Q - V  # [batch_size, tgt_len]
    
    # calculate V(s_t+1) + r_t - V(s_t)
    A_ = torch.zeros_like(Q)
    V_ = tau * (logits_ / tau).logsumexp(dim=-1)
    A_[:, :-1] = gamma * V_[:, 1:] - V_[:, :-1] + rewards[:, :-1]
    
    terminal_V_ = V_[
        torch.arange(seq_lens.shape[0]),
        seq_lens - 1]  # [batch_size]

    terminal_R = rewards[
        torch.arange(seq_lens.shape[0]),
        seq_lens - 1]

    A_[torch.arange(seq_lens.shape[0]),
       seq_lens - 1] = terminal_R - terminal_V_
    
    raw_losses = F.mse_loss(A, A_, reduction="none")
    return raw_losses


def multi_step_PCL_loss(
    logits, 
    logits_, 
    actions, 
    rewards, 
    seq_lens, 
    gamma=1.0, 
    tau=1.0
):
    """
    Multi-step unified path consistency learning (PCL). 
    
    See paper https://arxiv.org/pdf/2106.07704.pdf (Eq.17).

    Args:
        logits (Tensor): [batch_size, tgt_len, vocab_size]
        logits_ (Tensor): [batch_size, tgt_len, vocab_size]
        actions (Tensor): [batch_size, tgt_len]
        rewards (Tensor): [batch_size] or [batch_size, tgt_len]
        seq_lens (Tensor): [batch_size]
        gamma (float): the discount factor
        tau (float): Shannon entropy coefficient term
    
    """
    if rewards.dim() == 1:
        rewards_ = torch.zeros_like(actions).to(rewards)
        rewards_[torch.arange(seq_lens.shape[0]), 
                 seq_lens - 1] = rewards
        rewards = rewards_

    if logits.dim() == actions.dim() + 1:
        actions = actions.unsqueeze(-1)

    Q = logits.gather(dim=-1, index=actions).squeeze(-1)
    V = tau * (logits / tau).logsumexp(dim=-1)
    A = Q - V
    A2 = discounted_future_sum(
        A,
        seq_lens,
        gamma=gamma)

    V_ = tau * (logits_ / tau).logsumexp(dim=-1)
    R = discounted_future_sum(
        rewards,
        seq_lens,
        gamma=gamma)

    assert R.shape == V_.shape
    raw_losses = F.mse_loss(
        A2, R - V_,
        reduction="none")
    return raw_losses


def mixed_PCL_loss(
    logits, 
    logits_, 
    actions, 
    rewards, 
    seq_lens, 
    ignore_index=None, 
    reduce=True, 
    gamma=1.0, 
    tau=1.0
):
    """
    A mix of single- and multi-step PCL update.

    Args:
        logits (Tensor): [batch_size, tgt_length, vocab_size]
        logits_ (Tensor): [batch_size, tgt_length, vocab_size]
        actions (Tensor): [batch_size, tgt_length]
        rewards (Tensor): [batch_size] or [batch_size, tgt_length]
        seq_lens (Tensor): [batch_size]
        gamma (float): the discount factor
        tau (float): Shannon entropy coefficient term

    """
    s_pcl_loss = single_step_PCL_loss(
        logits, 
        logits_, 
        actions, 
        rewards, 
        seq_lens, 
        gamma=gamma, 
        tau=tau
    )
    m_pcl_loss = multi_step_PCL_loss(
        logits, 
        logits_, 
        actions, 
        rewards, 
        seq_lens, 
        gamma=gamma, 
        tau=tau
    )

    raw_losses = (s_pcl_loss + m_pcl_loss) / 2
    assert raw_losses.shape == actions.shape, \
        "Losses shape does not match: {}".format(raw_losses.shape)

    loss = mask_and_reduce(
        sequence=raw_losses,
        sequence_length=seq_lens,
        average_across_batch=True,
        average_across_timesteps=True,
        sum_over_batch=False,
        sum_over_timesteps=False
    )

    # mask & reduce
    # if ignore_index is not None:
    #     pad_mask = actions.eq(ignore_index)
    #     raw_losses.masked_fill_(pad_mask, 0.0)

    # if reduce:
    #     loss = raw_losses.mean()
    
    return loss


class SoftQLearningCriterion(nn.Module):
    def __init__(
        self,
        padding_idx,
        label_smoothing,
        report_accuracy=False,
        reward_shaping=False,
        old_r_min=0.,
        old_r_max=1.0,
        new_r_min=-0.5,
        new_r_max=0.5,
        gamma_pcl=1.0,
        tau_pcl=1.0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.eps = label_smoothing
        self.gamma = gamma_pcl
        self.tau = tau_pcl

        if reward_shaping:
            self._reward_shaping_func = get_reward_shaping_func(
                old_min=old_r_min,
                old_max=old_r_max,
                new_min=new_r_min,
                new_max=new_r_max)
        else:
            self._reward_shaping_func = lambda r: r

    def forward(self, logits, tgt_logits, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with two elements:
        1) the loss
        3) logging outputs to display while training

        Args:
            logits (Tensor): [batch_size, max_tgt_len, vocab_size]
            tgt_logits (Tensor): [batch_size, max_tgt_len, vocab_size]
            sample (dict): {
                'target' (Tensor): [batch_size, max_tgt_len]
                'rewards' (Tensor): [batch_size, max_tgt_len] or [batch_size]
            }
        """
        loss = self.compute_loss(
            logits, sample, tgt_logits=tgt_logits, reduce=reduce
        )

        logging_output = {
            "loss": loss.data,
            # "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
        }

        return loss, logging_output

    def compute_loss(
            self,
            logits,
            sample,
            tgt_logits=None,
            reduce=True
        ):
        """
        Args:
            logits (tuple):
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        target = sample['target']  # target: [batch_size, max_tgt_len]
        tgt_lengths = self.get_tgt_lengths(target)
        assert tgt_lengths.dim() == 1 and tgt_lengths.shape[0] == target.shape[0]

        print(target[0])
        print(tgt_lengths[0])

        rewards = None
        if sample.get('rewards', None) is not None:
            rewards = sample['rewards']
            assert rewards.shape[0] == target.shape[0] and \
                (rewards.dim() == 1 or rewards.shape[1] == target.shape[1]), \
                "Target size: {}; rewards size: {}.".format(target.size(), rewards.size())
        else:
            raise Exception("No rewards found!!!")

        rewards = self._reward_shaping_func(rewards)
        if rewards.dtype != logits[0].dtype and logits[0].dtype == torch.float16:
            rewards = rewards.half()

        loss = mixed_PCL_loss(
            logits,
            tgt_logits,
            target,
            rewards,
            tgt_lengths,
            ignore_index=self.padding_idx,
            reduce=reduce,
            gamma=self.gamma,
            tau=self.tau
        )
        return loss

    def get_tgt_lengths(self, target):
        """
        Args:
            target (torch.LongTensor): [batch_size, tgt_length]
            padding_idx: int
        """
        pad_mask = target.eq(self.padding_idx)
        return (~pad_mask).sum(dim=-1).to(target)