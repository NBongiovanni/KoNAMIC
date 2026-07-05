# KoNAMIC project notes

KoNAMIC is a robotics/control project for learning Koopman-based models from sensor or visual data and using them for closed-loop control.

Main architecture:
- `core/`: reusable systems, plants, generic controllers, simulation, and infrastructure.
- `koopman/`: Koopman-specific domain code.
  - `models/`: Koopman model composition, latent dynamics, model configs, checkpoints, and forward-output structures.
  - `lifting/`: lifting and reconstruction components such as encoders, decoders, MLPs, and auto-encoders.
  - `controllers/`: Koopman-specific controllers such as KMPC and KLQR.
  - `training/`: Koopman training loop, training context, checkpoint manager, curriculum, forward-loss computation, and losses.
- `pipelines/`: task-level orchestration for data generation, training, evaluation.
- `viz/`: plotting and visualization.

Supported systems:
- `quadrotor_2d`
- `quadrotor_3d`
- `cartpole`

Current focus:
- sensor-based Koopman learning for `quadrotor_2d`;
- state `x` is included in the lifted state `z`;
- first target model is linear Koopman: `z_next = A z + B u`;
- closed-loop augmentation uses KMPC/acados;
- bilinear Koopman may be considered later, especially for aerodynamic effects such as ground effect.

Coding preferences:
- avoid silent default values for required config parameters;
- prefer typed config objects over raw dictionaries when practical;
- keep evaluation and training-data generation conceptually separated;
- `TrainingEvaluator` should evaluate, while closed-loop augmentation should eventually be handled by a dedicated augmenter or runner;
- use explicit names such as `system_name`, `system_spec`, `plant`, `scenario_generator`.

Dependency boundaries:
- `core/` should not depend on `koopman/`;
- Koopman-specific code should live under `koopman/`, not under `core/`;
- `pipelines/` may orchestrate `core/` and `koopman/`, but should avoid owning Koopman-specific model, controller, or training primitives;
- keep configs and evaluators in `pipelines/` when they describe experiment orchestration rather than Koopman mechanics.

Design principle:
- keep simple, testable steps before adding bilinear models, NMPC, vision, or aerodynamic effects.
