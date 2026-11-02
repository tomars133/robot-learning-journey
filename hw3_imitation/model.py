# 과제 제공 스켈레톤(ETH Robot Learning 2026, HW3) 기반.
# 직접 구현: ObstaclePolicy, MultiTaskPolicy, build_policy의 인자 전달 부분.
"""SO-100 모방 정책을 위한 모델 정의."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """액션 청킹(action chunking) 정책의 기본 클래스."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """MSE 손실을 사용하여 액션 청크를 예측합니다.

    상태 벡터를 평탄화된 액션 청크(chunk_size * action_dim)로 매핑하고
    이를 (B, chunk_size, action_dim) 크기로 재구성하는 단순 MLP입니다.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """multicube 씬을 위한 목표 조건부(goal-conditioned) 정책."""
        raise NotImplementedError


# TODO: 학생들은 여기에 ObstaclePolicy를 구현하십시오.
class ObstaclePolicy(BasePolicy):
    """배치에 대한 학습 손실(training loss)을 계산합니다."""
    def __init__(self, state_dim, action_dim, chunk_size, d_model=128, depth=2):
        super().__init__(state_dim, action_dim, chunk_size)
        # (B, 10)의 state를 input으로 ->
        out_dim = self.action_dim * self.chunk_size # 액션을 청킹으로 내보낼거라서, 차원은 액션 차원 x 묶음수

        layers = []
        in_dim = state_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, d_model))
            layers.append(nn.GELU())
            in_dim = d_model
        layers.append(nn.Linear(d_model, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state) -> torch.Tensor:
        """(batch, chunk_size, action_dim) 크기의 액션 청크를 생성합니다."""
        out = self.net(state)
        return out.reshape(out.shape[0], self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        criterion = nn.MSELoss()
        preds = self.forward(state)
        loss = criterion(preds, action_chunk)
        return loss

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


# TODO: 학생들은 여기에 MultiTaskPolicy를 구현하십시오.
class MultiTaskPolicy(BasePolicy):
    """(B, chunk_size, action_dim) 크기의 예측된 액션 청크를 반환합니다."""
    def __init__(self, state_dim, action_dim, chunk_size, d_model=128, depth=2):
        super().__init__(state_dim, action_dim, chunk_size)
        assert state_dim == 19, f"state_dim이 19가 아님: {state_dim}"
        # 입력 : state (B, 19)에서 [4:7], [7:10], [10:13]은 각각 RGB 큐브의 위치, [13:16]은 목표 원핫
        # 목표 위치는 RGB 중 1개일 때의 위치만 남으므로, 목표 위치 shape은 (B, 3) -> input은 state의 1번째 축 [4:16]을 가공해서
        # shape이 (B, 10)이 됨.

        layers = []
        in_dim = 10
        out_dim = action_dim * chunk_size

        for _ in range(depth):
            layers.append(nn.Linear(in_dim, d_model))
            layers.append(nn.GELU())
            in_dim = d_model
        layers.append(nn.Linear(d_model, out_dim))
        self.net = nn.Sequential(*layers)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        pred = self.forward(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)

    def forward(self, state) -> torch.Tensor:
        """(B, chunk_size, action_dim) 크기의 예측된 액션 청크를 반환합니다."""
        ee = state[:, 0:3]
        gripper = state[:, 3:4]
        red = state[:, 4:7]
        green = state[:, 7:10]
        blue = state[:, 10:13]
        onehot = (state[:, 13:16] > 0).float()
        goal_pos = state[:, 16:19]
        target_pos = red*onehot[:, 0:1] + green*onehot[:, 1:2] + blue*onehot[:, 2:3]

        x = torch.cat([ee, gripper, target_pos, goal_pos],dim=-1)
        out = self.net(x)
        return out.reshape(out.shape[0], self.chunk_size, self.action_dim)

PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    # TODO,
    chunk_size: int,
    d_model: int = 128,
    depth: int = 2,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: 선택한 사양으로 빌드하십시오.
            chunk_size = chunk_size,
            d_model = d_model,
            depth = depth,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: 선택한 사양으로 빌드하십시오.
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
