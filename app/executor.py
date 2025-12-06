import sys
import io
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import base64
import traceback

# Set non-interactive backend to prevent plots from popping up on the server
matplotlib.use('Agg') 

class CodeExecutor:
    def __init__(self):
        # This dictionary acts as the "memory" of the session.
        # We initialize it with pandas so the code can always use 'pd'
        self.globals = {"pd": pd, "plt": plt}
        self.locals = {}

    def execute_code(self, code: str):
        """
        Executes Python code, captures stdout, and intercepts matplotlib plots.
        """
        # 1. Capture Standard Output (print statements)
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        image_base64 = None
        error_message = None

        try:
            # 2. Execute the code within the persistent context
            exec(code, self.globals, self.locals)

            # 3. Check if a plot was generated
            if plt.get_fignums():
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', bbox_inches='tight')
                img_buffer.seek(0)
                # Convert to base64 so we can send it as JSON
                image_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
                plt.close('all') # Clear plot for next time

        except Exception:
            # Capture the full traceback if code fails
            error_message = traceback.format_exc()
        
        finally:
            # Restore stdout
            sys.stdout = old_stdout

        return {
            "text_output": redirected_output.getvalue(),
            "image_output": image_base64,
            "error": error_message
        }

# Create a singleton instance for Phase 2 simplicity
# (In Phase 3/4, we would manage multiple sessions)
session_executor = CodeExecutor()