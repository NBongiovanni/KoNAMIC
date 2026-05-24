import argparse


def build_arg_parser_data_generation() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/Resume Koopman model (sensors)")
    p.add_argument("--drone_dim", type=int)
    return p