"""
Monkey patch for LiteLLM to support Langfuse v3.

LiteLLM currently adds 'sdk_integration' parameter for Langfuse >= 2.6.0,
but this parameter was removed in Langfuse v3.
"""

def patch_litellm_langfuse():
    """Apply monkey patch to fix Langfuse v3 compatibility."""
    try:
        import litellm.integrations.langfuse.langfuse as litellm_langfuse
        from packaging.version import Version
        
        # Store original __init__ method
        original_init = litellm_langfuse.LangFuseLogger.__init__
        
        def patched_init(self, **kwargs):
            # Call original init
            original_init(self, **kwargs)
            
            # Override the safe_init_langfuse_client to handle v3
            original_safe_init = self.safe_init_langfuse_client
            
            def patched_safe_init(parameters):
                # Remove sdk_integration for Langfuse v3+
                if "sdk_integration" in parameters:
                    import langfuse
                    langfuse_version = langfuse.version.__version__
                    if Version(langfuse_version) >= Version("3.0.0"):
                        parameters = parameters.copy()
                        parameters.pop("sdk_integration", None)
                
                return original_safe_init(parameters)
            
            self.safe_init_langfuse_client = patched_safe_init
        
        # Apply the patch
        litellm_langfuse.LangFuseLogger.__init__ = patched_init
        
        return True
    except Exception as e:
        print(f"Warning: Failed to apply Langfuse v3 patch: {e}")
        return False