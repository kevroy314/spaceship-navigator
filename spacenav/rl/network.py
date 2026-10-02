"""Actor-critic over the ego-centric set observation.

Bodies and zones are embedded token-by-token, then pooled two ways: multi-head
cross-attention with the ship's own features as the query ("which bodies matter
to me right now"), and a masked max-pool.  The pooled summary feeds shared
policy and value heads.  Nothing depends on the number or order of bodies, so
the same weights run on any level.
"""

import jax.numpy as jnp
import flax.linen as nn
import numpy as np

# Discrete action set = what a human can press: turn {left, none, right} x throttle {off, gentle, full}
TURNS = (1.0, 0.0, -1.0)
THROTTLES = (0.0, 0.3, 1.0)
ACTIONS = jnp.array([[t, g] for t in TURNS for g in THROTTLES], jnp.float32)
N_ACTIONS = ACTIONS.shape[0]


def _dense(n, scale=np.sqrt(2)):
    return nn.Dense(n, kernel_init=nn.initializers.orthogonal(scale), bias_init=nn.initializers.zeros)


class ActorCritic(nn.Module):
    d: int = 128
    heads: int = 4
    hidden: int = 256

    @nn.compact
    def __call__(self, obs):
        d, H = self.d, self.heads
        s = nn.relu(_dense(d)(obs["self"]))
        s = nn.relu(_dense(d)(s))

        def embed(x):
            return _dense(d)(nn.relu(_dense(d)(x)))

        tb = embed(obs["bodies"])
        tz = embed(obs["zones"])
        tokens = jnp.concatenate([tb, tz], axis=-2)                    # (..., T, d)
        mask = jnp.concatenate([obs["body_mask"], obs["zone_mask"]], axis=-1)

        dh = d // H
        q = _dense(d, 1.0)(s).reshape(s.shape[:-1] + (H, dh))
        k = _dense(d, 1.0)(tokens).reshape(tokens.shape[:-1] + (H, dh))
        v = _dense(d, 1.0)(tokens).reshape(tokens.shape[:-1] + (H, dh))
        att = jnp.einsum("...hd,...thd->...ht", q, k) / np.sqrt(dh)
        att = jnp.where(mask[..., None, :], att, -1e9)
        att = nn.softmax(att, axis=-1)
        pooled = jnp.einsum("...ht,...thd->...hd", att, v).reshape(s.shape[:-1] + (d,))
        maxpool = jnp.max(jnp.where(mask[..., None], nn.relu(tokens), 0.0), axis=-2)

        h = jnp.concatenate([s, pooled, maxpool], axis=-1)
        h = nn.relu(_dense(self.hidden)(h))
        h = nn.relu(_dense(self.hidden)(h))
        logits = _dense(N_ACTIONS, 0.01)(h)
        value = _dense(1, 1.0)(h)[..., 0]
        return logits, value
