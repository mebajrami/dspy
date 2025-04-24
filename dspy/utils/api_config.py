"""
Utilities for configuring DSPy to use custom API endpoints.
"""
from typing import Optional

def use_custom_api_endpoint(api_url: Optional[str] = None):
    """
    Configure DSPy to use the custom API endpoint.
    
    Args:
        api_url: URL for the custom API endpoint
        
    Returns:
        The previous adapter that was configured (for restoring later)
    """
    from dspy.dsp.utils.settings import settings
    from dspy.adapters.custom_api_adapter import CustomAPIAdapter
    
    # Save the current adapter to return
    previous_adapter = settings.adapter
    
    # Configure DSPy to use our custom adapter
    custom_adapter = CustomAPIAdapter(api_url=api_url)
    settings.configure(adapter=custom_adapter)
    
    return previous_adapter

def restore_adapter(adapter):
    """
    Restore a previously saved adapter configuration.
    
    Args:
        adapter: The adapter to restore (returned from use_custom_api_endpoint)
    """
    from dspy.dsp.utils.settings import settings
    settings.configure(adapter=adapter)