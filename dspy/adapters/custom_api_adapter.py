from typing import Any, Type, Optional
import requests
import logging

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback

logger = logging.getLogger(__name__)

class CustomAPIAdapter(ChatAdapter):
    """
    A custom adapter that redirects LLM calls to our new AutoOpt API endpoint.
    Formats requests and responses to be compatible with your existing API structure.
    """
    def __init__(
        self, 
        api_url: str = "http://localhost:4242/generativeai/classification/document/auto_opt",
        callbacks: Optional[list[BaseCallback]] = None
    ):
        super().__init__(callbacks)
        self.api_url = api_url
        logger.info(f"Initialized CustomAPIAdapter with endpoint: {self.api_url}")
    
    def __call__(
        self,
        lm,
        lm_kwargs: dict[str, Any],
        signature: Type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Override to redirect the call to our API endpoint
        """
        # Use parent class to format the inputs properly
        formatted_inputs = self.format(signature, demos, inputs)
        
        # Extract document text from inputs
        document_text = ""
        for field_name in ['document', 'text', 'context', 'content']:
            if field_name in inputs and isinstance(inputs[field_name], str):
                document_text = inputs[field_name]
                break
        
        # If we couldn't find a document field, use the first message content
        if not document_text and formatted_inputs:
            for msg in formatted_inputs:
                if msg.get('role') == 'user' and 'content' in msg:
                    document_text = msg['content']
                    break
        
        # Extract definition from signature instructions
        definition = signature.instructions if hasattr(signature, 'instructions') else "Classification task"
        
        # Prepare payload for your API endpoint in the format it expects
        payload = {
            "document": {
                "item_id": "dspy_document",
                "text": document_text
            },
            "definitions": [
                {
                    "definition_id": "dspy_call",
                    "definition": definition
                }
            ],
            "llm_config": lm_kwargs,
            "request_tracker": {"request_tracking": "DSPY_CUSTOM_ADAPTER"}
        }
        
        # For debugging
        logger.info(f"Sending request to custom endpoint: {self.api_url}")
        logger.info(f"Document length: {len(document_text)} chars")
        
        try:
            # Send request to your API
            response = requests.post(self.api_url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            logger.info(f"Response received: {result}")
            
            # First, check if the result contains a label field
            if "label" in result:
                # Convert to the format that DSPy expects - output text that can be parsed
                # The expected fields will depend on the signature's output fields
                output_text = {"text": f"label: {result['label']}"}
                if "confidence" in result:
                    output_text["text"] += f"\nconfidence: {result['confidence']}"
                
                return self._call_post_process([output_text], signature)
            else:
                # If there's no label, look for outputs field as a fallback
                outputs = result.get("outputs", [result.get("output", "")])
                return self._call_post_process(outputs, signature)
            
        except Exception as e:
            logger.error(f"Error calling custom API endpoint: {e}")
            # Return an error response in the expected format
            return [{field: f"Error: {e}" for field in signature.output_fields}]