from .training_evaluator import TrainingEvaluator
from .cli import parse_learning_args
from .config import (
    ClosedLoopTrainingConfig,
    LoggingConfig,
    OpenLoopTrainingConfig,
    PredictionHorizon,
    TrainerConfig,
    TrainingPipelineConfig,
    WandbLoggingConfig,
    save_effective_run_config
)


def __getattr__(name: str):
    if name == "Trainer":
        from KoNAMIC.koopman.training import Trainer

        return Trainer
    if name in {"TrainingContext", "build_training_context"}:
        from KoNAMIC.koopman.training import TrainingContext, build_training_context

        return {
            "TrainingContext": TrainingContext,
            "build_training_context": build_training_context,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
