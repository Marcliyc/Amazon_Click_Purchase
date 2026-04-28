import jax
import jax.numpy as jnp

from src.ev_beta_choice_cm import EVBetaChoiceCMConfig, constrained_params, init_for_optimizer, loss_fn, predicted_monthly_mean


def _cfg():
    return EVBetaChoiceCMConfig(
        amazon_fixed={
            "r": 0.457522,
            "alpha": 7.372648,
            "s": 16.440086,
            "beta": 16.369219,
            "r_v": 1.939664,
            "r_tau": 22.712252,
            "psi": -0.080496,
            "pi": 0.208507,
            "mu0": 0.751528,
            "k": 0.730525,
        },
        ebay_init={"pi": 0.208507, "mu0": 0.751528, "k": 0.730525},
        choice={"initial_mean": 0.5, "initial_concentration": 20.0, "fix_concentration": True, "min_concentration": 2.0},
        priors={"lambda_pi": 0.1, "lambda_mu": 0.1, "lambda_k": 0.1},
        fit={"likelihood": "negative_binomial", "obs_scale_init": 20.0},
    )


def test_transforms_and_initialization_match_amazon():
    cfg = _cfg()
    raw = init_for_optimizer(cfg)
    p = constrained_params(raw, cfg)
    assert p.pi_ebay > 0 and p.pi_ebay < 1
    assert p.mu0_ebay > 0
    assert p.k_ebay > 0
    assert abs(p.pi_ebay - p.pi_amazon) < 1e-5
    assert abs(p.mu0_ebay - p.mu0_amazon) < 1e-5
    assert abs(p.k_ebay - p.k_amazon) < 1e-5


def test_likelihood_prefers_close_prediction():
    cfg = _cfg()
    raw = init_for_optimizer(cfg)
    x = jnp.arange(4, dtype=jnp.float32)
    y = jnp.array([4.0, 6.0, 5.0, 7.0], dtype=jnp.float32)

    raw_good = dict(raw)
    raw_bad = dict(raw)
    raw_good["raw_pi_ebay"] = jnp.asarray(-1.0)
    raw_bad["raw_pi_ebay"] = jnp.asarray(-8.0)

    lg = float(loss_fn(raw_good, x, y, 50.0, cfg))
    lb = float(loss_fn(raw_bad, x, y, 50.0, cfg))
    assert lg < lb


def test_jit_grad_no_nans():
    cfg = _cfg()
    raw = init_for_optimizer(cfg)
    x = jnp.arange(3, dtype=jnp.float32)
    y = jnp.array([1.0, 1.0, 2.0], dtype=jnp.float32)

    wrapped = lambda rp, xx, yy, tc: loss_fn(rp, xx, yy, tc, cfg)
    jit_loss = jax.jit(wrapped)
    v = jit_loss(raw, x, y, 10.0)
    assert jnp.isfinite(v)

    grads = jax.grad(wrapped, argnums=0)(raw, x, y, 10.0)
    assert jnp.isfinite(grads["raw_pi_ebay"])

    pred = predicted_monthly_mean(raw, x, 10.0, cfg)
    assert jnp.all(pred > 0)
