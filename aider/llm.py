import importlib
import os
import warnings

from aider.dump import dump  # noqa: F401

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

AIDER_SITE_URL = "https://aider.chat"
AIDER_APP_NAME = "Aider"

os.environ["OR_SITE_URL"] = AIDER_SITE_URL
os.environ["OR_APP_NAME"] = AIDER_APP_NAME
os.environ["LITELLM_MODE"] = "PRODUCTION"

# `import litellm` takes 1.5 seconds, defer it!

VERBOSE = False


class LazyLiteLLM:
    _lazy_module = None

    def __getattr__(self, name):
        if name == "_lazy_module":
            return super()
        self._load_litellm()
        return getattr(self._lazy_module, name)

    def _load_litellm(self):
        if self._lazy_module is not None:
            return

        if VERBOSE:
            print("Loading litellm...")

        self._lazy_module = importlib.import_module("litellm")

        self._lazy_module.suppress_debug_info = True
        self._lazy_module.set_verbose = False
        self._lazy_module.drop_params = True
        self._lazy_module._logging._disable_debugging()
        
        # Configure Langfuse callbacks if any are pending
        self._configure_pending_langfuse_callbacks()
    
    def _configure_pending_langfuse_callbacks(self):
        """Configure any pending Langfuse callbacks from Coder instances."""
        # Check if langfuse should be enabled
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            self._lazy_module.success_callback = ["langfuse"]


litellm = LazyLiteLLM()


def configure_langfuse_callback(callback_handler):
    """Configure litellm to use a Langfuse callback handler."""
    if litellm._lazy_module is not None:
        # litellm is already loaded, configure it directly
        if not hasattr(litellm, 'callbacks'):
            litellm.callbacks = []
        if callback_handler not in litellm.callbacks:
            litellm.callbacks.append(callback_handler)
    else:
        # Store for later configuration when litellm loads
        # This is handled in _configure_pending_langfuse_callbacks
        pass


__all__ = [litellm]
