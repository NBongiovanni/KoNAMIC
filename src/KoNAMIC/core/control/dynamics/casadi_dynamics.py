from __future__ import annotations

import casadi as ca

from KoNAMIC.core.models import BaseKoopModel


def build_latent_dynamics_function(
        koop_model: BaseKoopModel,
        z_dim: int,
        u_dim: int,
        z_dynamics_model: str,
        *,
        augment_actuated: bool,   # <-- augmentation uniquement dans la partie actionnée
        augment_value: float = 1.0,
) -> ca.Function:
    z = ca.MX.sym("z", z_dim, 1)
    u = ca.MX.sym("u", u_dim, 1)

    if z_dynamics_model == "linear":
        A_t, B_t = koop_model.construct_koop_matrices()
        A = ca.DM(A_t.detach().cpu().numpy())
        B = ca.DM(B_t.detach().cpu().numpy())

        if augment_actuated:
            raise ValueError("augment_actuated is only meaningful for z_dynamics_model='bilinear'.")

        if A.size1() != z_dim or A.size2() != z_dim:
            raise ValueError(f"A must be ({z_dim},{z_dim}), got ({A.size1()},{A.size2()}).")
        if B.size1() != z_dim or B.size2() != u_dim:
            raise ValueError(f"B must be ({z_dim},{u_dim}), got ({B.size1()},{B.size2()}).")

        z_next = A @ z + B @ u
        return ca.Function("f_discrete", [z, u], [z_next])

    if z_dynamics_model != "bilinear":
        raise ValueError(f"Unknown z_dynamics_model={z_dynamics_model}")

    # bilinear
    a_mat, act_matrix = koop_model.construct_koop_matrices()
    A = ca.DM(a_mat.detach().cpu().numpy())
    Act = ca.DM(act_matrix.detach().cpu().numpy())

    if A.size1() != z_dim or A.size2() != z_dim:
        raise ValueError(f"For bilinear, A must be ({z_dim},{z_dim}), got ({A.size1()},{A.size2()}).")

    # Decide actuated feature dimension
    n_act = z_dim + 1 if augment_actuated else z_dim
    if augment_actuated:
        z_act = ca.vertcat(z, ca.MX(augment_value))  # (z_dim+1, 1)
    else:
        z_act = z  # (z_dim, 1)

    # act_matrix expected shape: (z_dim, u_dim * n_act)
    if Act.size1() != z_dim or Act.size2() != u_dim * n_act:
        raise ValueError(
            f"act_matrix must be ({z_dim},{u_dim*n_act}) with n_act={n_act}, "
            f"got ({Act.size1()},{Act.size2()})."
        )

    # z_{k+1} = A z + sum_i (B_i z_act) * u_i
    z_next = A @ z
    for i in range(u_dim):
        Bi = Act[:, i * n_act : (i + 1) * n_act]  # (z_dim, n_act)
        z_next += (Bi @ z_act) * u[i]

    return ca.Function("f_discrete", [z, u], [z_next])
