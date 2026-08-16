import gymnasium as gym
from gymnasium import spaces
import numpy as np


class VeiumtumEnv(gym.Env):
    def __init__(self, natural=False, sab=False):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2),
        ))
        self.natural = natural
        self.sab = sab

    def _draw_card(self):
        card = min(10, np.random.randint(1, 14))
        return card

    def _hand_value(self, hand):
        total = sum(hand)
        aces = hand.count(1)
        while total + 10 <= 21 and aces > 0:
            total += 10
            aces -= 1
        return total

    def _is_bust(self, hand):
        return self._hand_value(hand) > 21

    def _usable_ace(self, hand):
        return 1 if (1 in hand and self._hand_value(hand) <= 11) else 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card()]
        return self._get_obs(), {}

    def _get_obs(self):
        return (
            self._hand_value(self.player_hand),
            self.dealer_hand[0],
            self._usable_ace(self.player_hand),
        )

    def step(self, action):
        if action == 1:
            self.player_hand.append(self._draw_card())
            if self._is_bust(self.player_hand):
                return self._get_obs(), -1.0, True, False, {}

        while self._hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self._draw_card())

        player_val = self._hand_value(self.player_hand)
        dealer_val = self._hand_value(self.dealer_hand)

        if dealer_val > 21:
            reward = 1.0
        elif player_val > dealer_val:
            reward = 1.0
        elif player_val == dealer_val:
            reward = 0.0
        else:
            reward = -1.0

        if self.natural and len(self.player_hand) == 2 and player_val == 21:
            if len(self.dealer_hand) == 2 and dealer_val == 21:
                reward = 0.0
            else:
                reward = 1.5

        return self._get_obs(), reward, True, False, {}

    def render(self):
        print(f"Player: {self.player_hand} (sum={self._hand_value(self.player_hand)})")
        print(f"Dealer: [{self.dealer_hand[0]}, ?]")
