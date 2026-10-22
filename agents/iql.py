import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from agents.common import get_flat_batch
from utils.activation import get_activation
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP_CLASSES, GaussianActor, Value


class IQLAgent(flax.struct.PyTreeNode):
    """IQL: Implicit Q-Learning (Kostrikov et al. 2021).

    Offline RL algorithm that learns Q and V functions using only dataset actions,
    without requiring policy rollouts or OOD action queries. V is trained via
    expectile regression onto Q, enabling value learning for near-optimal policies.
    The actor is extracted by advantage-weighted regression (AWR) or DDPG+BC.
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
        """Compute the actor loss (AWR or DDPG+BC)."""
        batch_actions, _, _, _, valid_w = get_flat_batch(batch, self.config)
        if self.config["actor_loss"] == "awr":
            # AWR loss.
            v = self.network.select("value")(batch["observations"])
            q1, q2 = self.network.select(
                "target_critic" if self.config["target_extraction"] else "critic"
            )(batch["observations"], actions=batch_actions)
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config["alpha"])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select("actor")(
                batch["observations"], params=grad_params
            )
            log_prob = dist.log_prob(batch_actions)

            actor_loss = -(exp_a * log_prob * valid_w).mean()

            actor_info = {
                "actor_loss": actor_loss,
                "adv": adv.mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - batch_actions) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }

            return actor_loss, actor_info
        elif self.config["actor_loss"] == "ddpgbc":
            # DDPG+BC loss.
            dist = self.network.select("actor")(
                batch["observations"], params=grad_params
            )
            if self.config["const_std"]:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select("critic")(
                batch["observations"], actions=q_actions
            )
            q = jnp.minimum(q1, q2)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -(q * valid_w).mean() / jax.lax.stop_gradient(
                jnp.maximum((jnp.abs(q) * valid_w).mean(), 1e-8)
            )
            log_prob = dist.log_prob(batch_actions)

            bc_loss = -(self.config["alpha"] * log_prob * valid_w).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                "actor_loss": actor_loss,
                "q_loss": q_loss,
                "bc_loss": bc_loss,
                "q_mean": q.mean(),
                "q_abs_mean": jnp.abs(q).mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - batch_actions) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f"Unsupported actor loss: {self.config['actor_loss']}")

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

    @partial(jax.jit, static_argnames=["rejection_sampling"])
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
        rejection_sampling=1,
    ):
        """Sample actions from the actor."""
        dist = self.network.select("actor")(observations, temperature=temperature)
        actions = dist.sample(seed=seed, sample_shape=(rejection_sampling,))
        actions = jnp.clip(actions, -1, 1)
        q = self.network.select("critic")(
            jnp.repeat(observations[None], rejection_sampling, axis=0), actions=actions
        ).min(axis=0)
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
            encoders["actor"] = encoder_module()

        activation_fn = get_activation(config["activation"])
        print(f"Using activation function: {activation_fn}")

        # Define networks.
        value_def = Value(
            network_class=MLP_CLASSES[config["value_network_class"]],
            network_kwargs={
                **config["value_network_kwargs"],
                "activation": activation_fn,
            },
            num_ensembles=1,
            encoder=encoders.get("value"),
        )
        critic_def = Value(
            network_class=MLP_CLASSES[config["value_network_class"]],
            network_kwargs={
                **config["value_network_kwargs"],
                "activation": activation_fn,
            },
            num_ensembles=2,
            encoder=encoders.get("critic"),
        )
        actor_def = GaussianActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            activation=activation_fn,
            state_dependent_std=False,
            const_std=config["const_std"],
            encoder=encoders.get("actor"),
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_critic"] = params["modules_critic"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config(variant=None):
    """
    ogbench default config
    """
    config = ml_collections.ConfigDict(
        dict(
            agent_name="iql",
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
            activation="gelu",
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
            # IQL-specific hyperparameters.
            actor_loss="ddpgbc",  # Actor loss type: "awr" or "ddpgbc".
            alpha=1.0,  # Temperature in AWR or BC coefficient in DDPG+BC.
            target_extraction=True,  # Use target critic (True) or online critic (False) for AWR advantages.
            const_std=True,  # Use constant standard deviation for the Gaussian actor.
        )
    )
    return config


def get_exorl_config():
    config = get_config()
    config["batch_size"] = 1024
    config["actor_hidden_dims"] = (512, 512, 512)
    config["value_network_kwargs"]["hidden_dims"] = (512, 512, 512)
    config["actor_layer_norm"] = True
    config["activation"] = "mish"
    config["actor_loss"] = "awr"
    config["alpha"] = 3.0
    config["const_std"] = False
    config["dataset_action_clip_eps"] = None  # no clipping for exorl envs
    return config
