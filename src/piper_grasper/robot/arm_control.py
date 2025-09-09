import torch

def home(robot, entity_cfg, sim, state: dict, duration_s: float = 1.25, ema: float | None = 0.2):
    """
    Call this once per frame.
    - Initializes itself on first call (stores q0/q1 and timers in `state`)
    - Returns (done: bool, q_cmd: Tensor or None)
    """
    # init on first call
    if not state.get("active", False):
        q0 = robot.data.joint_pos[:, entity_cfg.joint_ids].clone()
        q1 = robot.data.default_joint_pos[:, entity_cfg.joint_ids].clone()
        state.clear()
        state.update({
            "active": True,
            "t": 0.0,
            "T": max(1e-3, float(duration_s)),
            "q0": q0, "q1": q1,
            "prev_q": None,
            "dt": float(sim.get_physics_dt()),
        })

    # progress one tick
    t  = min(state["t"] + state["dt"], state["T"])
    T  = state["T"]; q0 = state["q0"]; q1 = state["q1"]; pq = state["prev_q"]
    tau = torch.tensor(t / T, dtype=q0.dtype, device=q0.device)
    s = 10*tau**3 - 15*tau**4 + 6*tau**5     # minimum-jerk

    q = q0 + s * (q1 - q0)
    if ema is not None:
        if pq is None: pq = q
        q = ema * q + (1.0 - ema) * pq
        state["prev_q"] = q

    state["t"] = t
    done = bool(t >= T - 1e-9)
    if done: state["active"] = False

    return done, q
