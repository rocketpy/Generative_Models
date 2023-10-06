# minichatgpt - To Train ChatGPT In 5 Minutes with ColossalAI


# https://github.com/juncongmoo/minichatgpt

# pip install minichatgpt


# Usage
"""
The main entrypoint is Trainer. We only support PPO trainer now.
We support many training strategies:
NaiveStrategy: simplest strategy. Train on single GPU.
DDPStrategy: use torch.nn.parallel.DistributedDataParallel.
Train on multi GPUs.
ColossalAIStrategy: use Gemini and Zero of ColossalAI.
It eliminates model duplication on each GPU and supports offload.
It's very useful when training large models on multi GPUs.
"""

from copy import deepcopy
from colossalai.nn.optimizer import HybridAdam
from minichatgpt.trainer import PPOTrainer
from minichatgpt.trainer.strategies import ColossalAIStrategy
from minichatgpt.nn import GPTActor, GPTCritic, RewardModel


strategy = ColossalAIStrategy()

with strategy.model_init_context():
  # init your model here
  # load pretrained gpt2
  actor = GPTActor(pretrained='gpt2')
  critic = GPTCritic()
  initial_model = deepcopy(actor).cuda()
  reward_model = RewardModel(deepcopy(critic.model), deepcopy(critic.value_head)).cuda()

actor_optim = HybridAdam(actor.parameters(), lr=5e-6)
critic_optim = HybridAdam(critic.parameters(), lr=5e-6)

# prepare models and optimizers
(actor, actor_optim), (critic, critic_optim), reward_model, initial_model = strategy.prepare(
        (actor, actor_optim), (critic, critic_optim), reward_model, initial_model)

# load saved model checkpoint after preparing
strategy.load_model(actor, 'actor_checkpoint.pt', strict=False)
# load saved optimizer checkpoint after preparing
strategy.load_optimizer(actor_optim, 'actor_optim_checkpoint.pt')

trainer = PPOTrainer(strategy,
                     actor,
                     critic,
                     reward_model,
                     initial_model,
                     actor_optim,
                     critic_optim,
                     ...)

trainer.fit(dataset, ...)

# save model checkpoint after fitting on only rank0
strategy.save_model(actor, 'actor_checkpoint.pt', only_rank0=True)
# save optimizer checkpoint on all ranks
strategy.save_optimizer(actor_optim, 'actor_optim_checkpoint.pt', only_rank0=False)
