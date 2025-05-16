from typing import Any, Dict, List

def convert_to_json_serializable(obj):
    """
    Convert Pydantic models and other objects to JSON serializable format.
    
    Args:
        obj: The object to convert
        
    Returns:
        JSON serializable version of the object
    """
    if hasattr(obj, "model_dump"):  # Pydantic v2 method
        return obj.model_dump()
    elif hasattr(obj, "dict"):      # Pydantic v1 method
        return obj.dict()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    else:
        return obj