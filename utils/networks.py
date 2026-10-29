import math
from dataclasses import field
from typing import Any, Callable, Dict, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={"params": 0, "intermediates": 0},
        split_rngs={"params": True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activation: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activation: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False
    dtype: Any = None
    layer_norm_affine: bool = False  # Only controls adaptive LayerNorm.

    @nn.compact
    def __call__(self, x, condition_features=None):
        if condition_features is not None and not self.layer_norm:
            raise ValueError("Conditioned inputs require layer_norm=True.")
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(
                size,
                kernel_init=self.kernel_init,
                dtype=self.dtype,
            )(x)
            if condition_features is not None:
                x = _adaptive_layer_norm(
                    x,
                    condition_features,
                    layer_norm_affine=self.layer_norm_affine,
                    dtype=self.dtype,
                )
            elif i + 1 < len(self.hidden_dims) and self.layer_norm:
                x = nn.LayerNorm(dtype=self.dtype)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activation(x)
            if i == len(self.hidden_dims) - 2:
                self.sow("intermediates", "feature", x)
        return x


def _adaptive_layer_norm(x, condition_features, *, layer_norm_affine, dtype):
    """Normalize features, then apply condition-dependent scale and shift.
    layer_norm_affine: False corresponds to AdaLN, True corresponds to FiLM.
    """
    x = nn.LayerNorm(
        use_scale=layer_norm_affine, use_bias=layer_norm_affine, dtype=dtype
    )(x)
    scale_shift = nn.Dense(
        2 * x.shape[-1],
        kernel_init=nn.initializers.zeros_init(),
        bias_init=nn.initializers.zeros_init(),
        dtype=dtype,
    )(condition_features)
    scale, shift = jnp.split(scale_shift, 2, axis=-1)
    return x * (1.0 + scale) + shift


class ResidualBlock(nn.Module):
    """Residual dense block, optionally using AdaLN from encoded condition features."""

    hidden_dim: int
    activation: Any = nn.gelu
    kernel_init: Any = default_init()
    layer_norm: bool = True
    dtype: Any = None
    layer_norm_affine: bool = False  # Only controls adaptive LayerNorm.

    @nn.compact
    def __call__(self, x, condition_features=None):
        if condition_features is not None and not self.layer_norm:
            raise ValueError("Conditioned inputs require layer_norm=True.")
        residual = x
        x = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init, dtype=self.dtype)(x)
        x = self.activation(x)
        if condition_features is not None:
            x = _adaptive_layer_norm(
                x,
                condition_features,
                layer_norm_affine=self.layer_norm_affine,
                dtype=self.dtype,
            )
        elif self.layer_norm:
            x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init, dtype=self.dtype)(x)
        x = residual + x
        x = self.activation(x)
        if condition_features is not None:
            x = _adaptive_layer_norm(
                x,
                condition_features,
                layer_norm_affine=self.layer_norm_affine,
                dtype=self.dtype,
            )
        elif self.layer_norm:
            x = nn.LayerNorm(dtype=self.dtype)(x)
        return x


class ResidualMLP(nn.Module):
    """OCBench residual MLP with two hidden-width entries per block and optional AdaLN."""

    hidden_dims: Sequence[int]
    output_dim: Optional[int] = None
    activation: Any = nn.gelu
    kernel_init: Any = default_init()
    layer_norm: bool = True
    dtype: Any = None
    layer_norm_affine: bool = False  # Only controls adaptive LayerNorm.

    @nn.compact
    def __call__(self, x, condition_features=None):
        if condition_features is not None and not self.layer_norm:
            raise ValueError("Conditioned inputs require layer_norm=True.")
        if len(self.hidden_dims) % 2:
            raise ValueError(
                f"ResidualMLP requires an even number of hidden layers; got {len(self.hidden_dims)}."
            )
        hidden_dim = self.hidden_dims[0]
        if any(dim != hidden_dim for dim in self.hidden_dims):
            raise ValueError(
                f"ResidualMLP requires constant hidden widths; got {self.hidden_dims}."
            )

        x = nn.Dense(hidden_dim, kernel_init=self.kernel_init, dtype=self.dtype)(x)
        x = self.activation(x)
        if self.layer_norm:
            x = nn.LayerNorm(dtype=self.dtype)(x)

        for _ in range(len(self.hidden_dims) // 2):
            x = ResidualBlock(
                hidden_dim=hidden_dim,
                activation=self.activation,
                kernel_init=self.kernel_init,
                layer_norm=self.layer_norm,
                dtype=self.dtype,
                layer_norm_affine=self.layer_norm_affine,
            )(x, condition_features)

        if self.output_dim is not None:
            x = nn.Dense(
                self.output_dim, kernel_init=self.kernel_init, dtype=self.dtype
            )(x)
        return x


class BroNet(nn.Module):
    """
    BroNet for critic learning: https://arxiv.org/pdf/2405.16158
    It's one architecture (among others) we can use to scale up critic networks.

    It's basically residual connections + layer normalization.
    """

    num_blocks: int
    hidden_dim: int
    kernel_init: Callable[[jnp.ndarray], jnp.ndarray] = default_init()
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)
        for _ in range(self.num_blocks):
            res = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
            res = nn.LayerNorm()(res)
            res = self.activation(res)
            res = nn.Dense(self.hidden_dim, kernel_init=default_init())(res)
            res = nn.LayerNorm()(res)
            x += res
        x = nn.Dense(1, kernel_init=default_init())(x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param(
            "log_value", init_fn=lambda key: jnp.full((), jnp.log(self.init_value))
        )
        return jnp.exp(log_value)


MLP_CLASSES = {
    "bro": BroNet,
    "mlp": MLP,
    "resmlp": ResidualMLP,
}


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


def timestep_embedding(t, emb_size=16, max_period=10000):
    """t is between [0, 1]."""
    t = jax.lax.convert_element_type(t, jnp.float32)
    t = t * max_period
    dim = emb_size
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
    )
    args = t[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return embedding


def embed_time(t, time_embedding: str):
    """Embed or pass through flow/diffusion time inputs."""
    if time_embedding == "sinusoidal":
        return timestep_embedding(t)
    if time_embedding == "raw":
        t = jax.lax.convert_element_type(t, jnp.float32)
        if t.ndim == 1:
            return t[..., None]
        return t
    raise ValueError(
        f"Unknown time_embedding {time_embedding!r}; expected 'sinusoidal' or 'raw'."
    )


class ActorFlowField(nn.Module):
    """Flow / diffusion policy.

    Attributes:
        hidden_dims: Hidden layer dimensions for the trunk MLP.
        action_dim: Action dimension.
        mlp_kwargs: Keyword arguments forwarded to the trunk MLP.
        encoder: Optional encoder for observations.
        time_embedding: ``'sinusoidal'`` (default) or ``'raw'`` concatenation.
    """

    hidden_dims: Any
    action_dim: int
    mlp_kwargs: dict = flax.struct.field(pytree_node=False)
    encoder: nn.Module = None
    time_embedding: str = "sinusoidal"
    mlp_class: Any = MLP

    @nn.compact
    def __call__(self, obs, noised_action, t=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            obs = self.encoder(obs)

        parts = [obs, noised_action]
        if t is not None:
            parts.append(embed_time(t, self.time_embedding))

        concat_input = jnp.concatenate(parts, axis=-1)
        final_activation_kwargs = (
            {} if self.mlp_class is ResidualMLP else {"activate_final": True}
        )
        outputs = self.mlp_class(
            self.hidden_dims, **self.mlp_kwargs, **final_activation_kwargs
        )(concat_input)
        v = nn.Dense(self.action_dim)(outputs)
        return v


class BinaryConditionalFlowField(nn.Module):
    """
    Input is a noised action, observation, and t. is_positive is an integer class
    index that is embedded. Used for CFGRL.
    """

    hidden_dims: Any
    action_dim: int
    mlp_kwargs: dict = flax.struct.field(pytree_node=False)
    mlp_class: Any = MLP

    @nn.compact
    def __call__(self, obs, is_positive, noised_action, t):
        t_embedding = timestep_embedding(t)
        is_positive_embedding = nn.Embed(2, 32)(is_positive)
        concat_input = jnp.concatenate(
            [obs, noised_action, t_embedding, is_positive_embedding], axis=-1
        )
        final_activation_kwargs = (
            {} if self.mlp_class is ResidualMLP else {"activate_final": True}
        )
        outputs = self.mlp_class(
            self.hidden_dims, **self.mlp_kwargs, **final_activation_kwargs
        )(concat_input)
        v = nn.Dense(self.action_dim, name="Output")(outputs)
        return v


class ConditionedMLP(nn.Module):
    """Encode a scalar condition and inject it through an MLP or ResidualMLP trunk."""

    hidden_dims: Sequence[int]
    activation: Any = nn.gelu
    layer_norm: bool = False
    layer_norm_affine: bool = False
    condition_hidden_dims: Sequence[int] = (128, 128)
    dtype: Any = None
    mlp_class: Any = MLP

    def setup(self):
        self.condition_encoder = MLP(
            self.condition_hidden_dims,
            activation=self.activation,
            activate_final=True,
            dtype=self.dtype,
            name="MLP_0",
        )
        mlp_kwargs = dict(
            activation=self.activation,
            layer_norm=self.layer_norm,
            layer_norm_affine=self.layer_norm_affine,
            dtype=self.dtype,
        )
        if self.mlp_class is MLP:
            mlp_kwargs.update(
                activate_final=True,
                kernel_init=nn.initializers.lecun_normal(),
            )
        self.mlp = self.mlp_class(self.hidden_dims, **mlp_kwargs)

    def __call__(self, x, condition):
        condition_features = self.condition_encoder(condition[..., None])
        return self.mlp(x, condition_features)

    def paired(self, x, condition, ref_condition):
        """Evaluate both conditions in one batch, broadcasting a scalar reference."""
        batch_size = x.shape[0]
        conditions = jnp.concatenate([condition, ref_condition], axis=0)
        encoded = self.condition_encoder(conditions[..., None])
        reference_features = jnp.broadcast_to(
            encoded[batch_size:], encoded[:batch_size].shape
        )
        condition_features = jnp.concatenate(
            [encoded[:batch_size], reference_features], axis=0
        )
        features = self.mlp(jnp.concatenate([x, x], axis=0), condition_features)
        return features[:batch_size], features[batch_size:]


class ContinuousConditionedFlowField(nn.Module):
    """Flow policy with a scalar condition modulating every hidden layer.
    This is used by RCP agents for example
    """

    hidden_dims: Sequence[int]
    action_dim: int
    mlp_kwargs: dict = flax.struct.field(pytree_node=False)
    condition_hidden_dims: Sequence[int] = (128, 128)
    layer_norm_affine: bool = False
    mlp_class: Any = MLP
    dtype: Any = None

    @nn.compact
    def __call__(self, obs, condition, noised_action, t):
        inputs = jnp.concatenate([obs, noised_action, timestep_embedding(t)], axis=-1)
        features = ConditionedMLP(
            hidden_dims=self.hidden_dims,
            activation=self.mlp_kwargs.get("activation", nn.gelu),
            layer_norm=self.mlp_kwargs.get("layer_norm", False),
            layer_norm_affine=self.layer_norm_affine,
            condition_hidden_dims=self.condition_hidden_dims,
            dtype=self.dtype,
            mlp_class=self.mlp_class,
            name="FiLMConditionedMLP_0",
        )(inputs, condition)
        return nn.Dense(self.action_dim, dtype=jnp.float32)(features)


class ConditionalAndUncondFlowField(nn.Module):
    """advantage-conditioned flow field (and unconditional BC flow field) for Advantage Guidance.

    The conditional branch uses AdaLN for the conditional advantage inputs.
    With residual parameterization, its output is stop_gradient(v_bc) + h(adv).
    Otherwise, h(adv) directly predicts the conditional velocity.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    mlp_kwargs: dict = flax.struct.field(pytree_node=False)
    condition_hidden_dims: Sequence[int] = (128, 128)
    residual_parameterization: bool = True
    mlp_class: Any = MLP
    dtype: Any = None

    def setup(self):
        # set up unconditional actor
        final_activation_kwargs = (
            {} if self.mlp_class is ResidualMLP else {"activate_final": True}
        )  # ResidualMLP always uses activations
        self.unconditional_mlp = self.mlp_class(
            self.hidden_dims,
            dtype=self.dtype,
            **self.mlp_kwargs,
            **final_activation_kwargs,
            name="unconditional_mlp",
        )
        self.unconditional_output = nn.Dense(
            self.action_dim, dtype=jnp.float32, name="unconditional_output"
        )  # keep output head in FP32 always
        # set up conditional actor
        self.conditional_mlp = ConditionedMLP(
            self.hidden_dims,
            activation=self.mlp_kwargs.get("activation", nn.gelu),
            layer_norm=self.mlp_kwargs.get("layer_norm", False),
            layer_norm_affine=False,
            condition_hidden_dims=self.condition_hidden_dims,
            mlp_class=self.mlp_class,
            dtype=self.dtype,
            name="conditional_mlp",
        )
        self.conditional_output = nn.Dense(
            self.action_dim, dtype=jnp.float32, name="conditional_output"
        )  # keep output head in FP32 always

    def _common_inputs(self, obs, noised_action, t):
        return jnp.concatenate([obs, noised_action, timestep_embedding(t)], axis=-1)

    def unconditional(self, obs, noised_action, t):
        """Evaluate the BC field for training, imagined actions, and inference."""
        inputs = self._common_inputs(obs, noised_action, t)
        return self.unconditional_output(self.unconditional_mlp(inputs))

    def __call__(self, obs, advantage, noised_action, t):
        """Return conditional and BC velocities, stopping BC gradients in the residual."""
        inputs = self._common_inputs(obs, noised_action, t)
        unconditional_velocity = self.unconditional_output(
            self.unconditional_mlp(inputs)
        )
        conditional_velocity = self.conditional_output(
            self.conditional_mlp(inputs, advantage)
        )
        if self.residual_parameterization:
            conditional_velocity = (
                jax.lax.stop_gradient(unconditional_velocity) + conditional_velocity
            )
        return conditional_velocity, unconditional_velocity

    def conditional_contrast(self, obs, positive_adv, negative_adv, noised_action, t):
        """Compute v_cond(positive_adv) - v_cond(negative_adv) for contrastive guidance."""
        inputs = self._common_inputs(obs, noised_action, t)
        positive_features, negative_features = self.conditional_mlp.paired(
            inputs, positive_adv, negative_adv
        )
        # We compute the output difference more efficiently here without calling self.conditional_output() twice.
        # For Dense(h) = h @ W + b, Dense(h_pos) - Dense(h_neg) = (h_pos - h_neg) @ W.
        # Project the feature difference once, then subtract Dense(0) = b to remove
        # the bias introduced by Dense(h_pos - h_neg).
        feature_difference = positive_features - negative_features
        output_bias = self.conditional_output(
            jnp.zeros((1, feature_difference.shape[-1]), dtype=feature_difference.dtype)
        )
        return self.conditional_output(feature_difference) - output_bias


class GaussianActor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        activation: Activation function (default: nn.gelu).
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    activation: Any = nn.gelu
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm,
            activation=self.activation,
        )
        self.mean_net = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )
        else:
            if not self.const_std:
                self.log_stds = self.param(
                    "log_stds", nn.initializers.zeros, (self.action_dim,)
                )

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        network_class: Network class (MLP, ResidualMLP, or BroNet).
        network_kwargs: Keyword arguments for the network architecture.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    num_ensembles: int = 2
    encoder: nn.Module = None
    network_class: type[nn.Module] = MLP
    network_kwargs: Dict[str, Any] = field(default_factory=dict)

    def setup(self):
        network = self.network_class
        if network is MLP:
            network_args = {
                "hidden_dims": (*self.network_kwargs["hidden_dims"], 1),
                "activate_final": False,
                "layer_norm": self.network_kwargs["layer_norm"],
            }
            if "activation" in self.network_kwargs:
                network_args["activation"] = self.network_kwargs["activation"]
        elif network is ResidualMLP:
            network_args = {
                "hidden_dims": self.network_kwargs["hidden_dims"],
                "output_dim": 1,
                "layer_norm": self.network_kwargs["layer_norm"],
            }
            if "activation" in self.network_kwargs:
                network_args["activation"] = self.network_kwargs["activation"]
        elif network is BroNet:
            network_args = {
                "num_blocks": self.network_kwargs["num_blocks"],
                "hidden_dim": self.network_kwargs["hidden_dim"],
            }
        else:
            raise ValueError(f"Invalid network class: {self.network_class}")

        if self.num_ensembles > 1:
            network = ensemblize(network, self.num_ensembles)
        value_net = network(**network_args)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v
