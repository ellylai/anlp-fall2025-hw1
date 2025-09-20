from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        max_grad_norm: float = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # TODO: Clip gradients if max_grad_norm is set
            if group["max_grad_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    group["params"], max_norm=group["max_grad_norm"]
                )
                # raise NotImplementedError()

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                # raise NotImplementedError()

                # State should be stored in this dictionary
                state = self.state[p]

                # TODO: Access hyperparameters from the `group` dictionary
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)  # first moment
                    state["exp_avg_sq"] = torch.zeros_like(p.data)  # second moment

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                alpha = group["lr"]
                beta1, beta2 = group["betas"]

                state["step"] += 1  # increment step counter: t = t+1

                # TODO: Update first and second moments of the gradients
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                v.mul_(beta2).add_(grad**2, alpha=1 - beta2)

                # TODO: Bias correction
                # Please note that we are using the "efficient version" given in Algorithm 2
                # https://arxiv.org/pdf/1711.05101

                bias_correction1 = 1.0
                bias_correction2 = 1.0
                if group["correct_bias"]:
                    bias_correction1 = 1 - beta1 ** state["step"]  # 1 - beta1^t
                    bias_correction2 = 1 - beta2 ** state["step"]  # 1 - beta2^t

                # TODO: Update parameters
                step_size = group["lr"] / bias_correction1
                # sqrt(v_hat) + eps
                denom = (v / bias_correction2).sqrt().add_(group["eps"])
                # Final parameter update
                # theta += -step_size * (m / denom); no weight decay YET
                p.data.addcdiv_(m, denom, value=-step_size)
                # --> theta_t = theta_{t-1} - step_size * ( m / (sqrt(v) + eps) + theta_{t-1})

                # TODO: Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                if group["weight_decay"] != 0:
                    # theta *= (1 - alpha*weight_decay)
                    p.data.mul_(1.0 - alpha * group["weight_decay"])

        return loss
