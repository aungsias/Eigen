from ..env import TradeEnv
from ..tools import strftime

from stable_baselines3.a2c import A2C
from stable_baselines3.common.base_class import BaseAlgorithm

def trade(env: TradeEnv, model: BaseAlgorithm, preds: list, progress: bool=True):
    state, _ = env.reset()
    done = False

    while not done:
        actions, _ = model.predict(state)
        state, _, done, _, _ = env.step(actions)

    equity = env.equity_memory
    balance = equity.iloc[-1, 0]
    preds.append(equity)

    if progress:
        print(f"{strftime(env.data.index.max())} balance: ${balance:,.2f}")

    return equity, balance, preds

def train_A2C(env, n_eps=10, verbose=0, progress_bar=False):
    timesteps = len(env.data) * n_eps
    model = A2C("MlpPolicy", env, verbose=verbose)
    model.learn(total_timesteps=timesteps, progress_bar=progress_bar)
    return model