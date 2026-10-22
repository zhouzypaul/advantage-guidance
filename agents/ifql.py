import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from agents.common import get_flat_batch
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP_CLASSES, ActorFlowField, Value


class IFQLAgent(flax.struct.PyTreeNode):
    """IFQL: Implicit Flow Q-Learning, the flow-matching variant of IDQL (Hansen-Estruch et al. 2023).

    Uses an IQL critic and performs implicit policy extraction: samples multiple
    candidate actions from the BC flow policy and selects the one with highest Q-value
    (best-of-N). The flow policy is trained with BC only; no RL actor updates.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        batch_actions, _, _, _, valid_w = get_flat_batch(batch, self.config)
        q1, q2 = self.network.select("target_critic")(
            batch["observations"], actions=batch_actions
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select("value")(batch["observations"], params=grad_params)
        value_loss = (
            self.expectile_loss(q - v, q - v, self.config["expectile"]) * valid_w
        ).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss."""
        batch_actions, next_obs, rewards, masks, valid_w = get_flat_batch(
            batch, self.config
        )
        next_v = self.network.select("value")(next_obs)
        q = (
            rewards
            + self.config["discount"] ** self.config["horizon_length"] * masks * next_v
        )

        q1, q2 = self.network.select("critic")(
            batch["observations"], actions=batch_actions, params=grad_params
        )
        critic_loss = (((q1 - q) ** 2 + (q2 - q) ** 2) * valid_w).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the behavioral flow-matching actor loss."""
        batch_actions, _, _, _, valid_w = get_flat_batch(batch, self.config)
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_flow")(
            batch["observations"], x_t, t, params=grad_params
        )
        actor_loss = (((pred - vel) ** 2).mean(axis=-1) * valid_w).mean()

        return actor_loss, {
            "actor_loss": actor_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
    ):
        """Sample actions from the actor."""
        orig_observations = observations
        if self.config["encoder"] is not None:
            observations = self.network.select("actor_flow_encoder")(observations)
        action_seed, _ = jax.random.split(seed)

        # Sample `num_samples` noises and propagate them through the flow.
        actions = jax.random.normal(
            action_seed,
            (
                self.config["num_samples"],
                *observations.shape[:-1],
                self.config["action_dim"]
                * (
                    self.config["horizon_length"]
                    if self.config["action_chunking"]
                    else 1
                ),
            ),
        )
        n_observations = jnp.repeat(
            jnp.expand_dims(observations, 0), self.config["num_samples"], axis=0
        )
        n_orig_observations = jnp.repeat(
            jnp.expand_dims(orig_observations, 0), self.config["num_samples"], axis=0
        )
        for i in range(self.config["flow_steps"]):
            t = jnp.full(
                (self.config["num_samples"], *observations.shape[:-1], 1),
                i / self.config["flow_steps"],
            )
            vels = self.network.select("actor_flow")(
                n_observations, actions, t, is_encoded=True
            )
            actions = actions + vels / self.config["flow_steps"]
        actions = jnp.clip(actions, -1, 1)

        # Pick the action with the highest Q-value.
        q = self.network.select("critic")(n_orig_observations, actions=actions).min(
            axis=0
        )
        actions = jnp.take_along_axis(
            actions, jnp.argmax(q, axis=0)[None, ..., None], axis=0
        )[0]
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            ex_actions = jnp.concatenate(
                [ex_actions] * config["horizon_length"], axis=-1
            )
        full_action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["value"] = encoder_module()
            encoders["critic"] = encoder_module()
            encoders["actor_flow"] = encoder_module()

        # Define networks.
        value_def = Value(
            network_class=MLP_CLASSES[config["value_network_class"]],
            network_kwargs=config["value_network_kwargs"],
            num_ensembles=1,
            encoder=encoders.get("value"),
        )
        critic_def = Value(
            network_class=MLP_CLASSES[config["value_network_class"]],
            network_kwargs=config["value_network_kwargs"],
            num_ensembles=2,
            encoder=encoders.get("critic"),
        )
        actor_flow_def = ActorFlowField(
            config["actor_hidden_dims"],
            full_action_dim,
            mlp_kwargs=dict(layer_norm=config["actor_layer_norm"]),
            encoder=encoders.get("actor_flow"),
            time_embedding="raw",
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times)),
        )
        if encoders.get("actor_flow") is not None:
            # Add actor_flow_encoder to ModuleDict to make it separately callable.
            network_info["actor_flow_encoder"] = (
                encoders.get("actor_flow"),
                (ex_observations,),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_critic"] = params["modules_critic"]

        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="ifql",
            action_dim=ml_collections.config_dict.placeholder(
                int
            ),  # Set automatically.
            # Common hyperparameters.
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            value_network_class="mlp",
            value_network_kwargs=dict(
                hidden_dims=(512, 512, 512, 512),
                layer_norm=True,
            ),
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            # n-step returns and action chunking.
            horizon_length=1,
            action_chunking=False,
            # RL hyperparameters.
            discount=0.99,
            tau=0.005,
            expectile=0.9,  # IQL expectile.
            flow_steps=10,
            # IFQL-specific hyperparameters.
            num_samples=32,  # Number of action samples for rejection sampling in action selection.
        )
    )
    return config
