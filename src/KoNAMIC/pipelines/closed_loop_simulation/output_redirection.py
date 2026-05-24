from pathlib import Path
from contextlib import contextmanager
import sys

@contextmanager
def redirect_output_to_file(file_path: Path, also_print: bool = True):
    """
    Context manager to redirect stdout/stderr to file.

    Provides Unix 'tee'-like functionality: output goes to both
    terminal and file simultaneously (if also_print=True).

    Args:
        file_path: Path to the output file
        also_print: If True, also print to terminal (default: True)

    Example:
        with redirect_output_to_file(Path("output.log")):
            print("This goes to both terminal and file")
    """

    class Tee:
        """Write to multiple streams simultaneously."""

        def __init__(self, file, terminal=None):
            self.file = file
            self.terminal = terminal

        def write(self, message):
            self.file.write(message)
            self.file.flush()  # Immediate write
            if self.terminal:
                self.terminal.write(message)
                self.terminal.flush()

        def flush(self):
            self.file.flush()
            if self.terminal:
                self.terminal.flush()

    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Open file
    log_file = open(file_path, 'w', encoding='utf-8')

    try:
        if also_print:
            # Tee mode: write to both file and terminal
            sys.stdout = Tee(log_file, original_stdout)
            sys.stderr = Tee(log_file, original_stderr)
        else:
            # Silent mode: write only to file
            sys.stdout = log_file
            sys.stderr = log_file

        print(f"[INFO] Output logging to: {file_path}")
        yield log_file

    finally:
        # Always restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
