# KoNAMIC project notes

KoNAMIC is a robotics/control project for learning Koopman-based models from sensor or visual data and using them for closed-loop control.

Main architecture:
- `core/`: reusable systems, plants, controllers, simulation, models.
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

Design principle:
- keep simple, testable steps before adding bilinear models, NMPC, vision, or aerodynamic effects.