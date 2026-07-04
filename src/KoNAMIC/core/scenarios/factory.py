from KoNAMIC.core.systems import SystemSpec, DroneSpec, CartPoleSpec
from KoNAMIC.core.scenarios.config import ScenarioGenerationConfig
from KoNAMIC.core.scenarios.scenario_generator import ScenarioGenerator
from KoNAMIC.core.scenarios.generators.quadrotor_2d import Quadrotor2DScenarioGenerator
from KoNAMIC.core.scenarios.generators.quadrotor_3d import Quadrotor3DScenarioGenerator
from KoNAMIC.core.scenarios.generators.cartpole import CartPoleScenarioGenerator


def build_scenario_generator(
    system_spec: SystemSpec,
    cfg: ScenarioGenerationConfig,
    seed: int | None = None,
) -> ScenarioGenerator:

    if isinstance(system_spec, DroneSpec):
        if system_spec.system_dim == 2:
            return Quadrotor2DScenarioGenerator(
                system=system_spec,
                cfg=cfg,
                seed=seed,
            )
        elif system_spec.system_dim == 3:
            return Quadrotor3DScenarioGenerator(
                system=system_spec,
                cfg=cfg,
                seed=seed,
            )
        raise ValueError(f"Unsupported quadrotor sys_dim={system_spec.system_dim}")

    if isinstance(system_spec, CartPoleSpec):
        return CartPoleScenarioGenerator(
            system=system_spec,
            cfg=cfg,
            seed=seed,
        )

    raise TypeError(
        f"Unsupported system specification type: {type(system_spec).__name__}"
    )