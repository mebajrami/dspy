from typing import Any, Type, Optional
import requests
import logging
import re
import os
import json

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback

logger = logging.getLogger(__name__)

class CustomAPIAdapter(ChatAdapter):
    """
    A custom adapter that redirects LLM calls to a configurable API endpoint.
    Formats requests and responses to be compatible with your existing API structure.
    """
    def __init__(
        self, 
        api_url: Optional[str] = None,
        instruction_url: Optional[str] = None,  # New parameter for instruction generation endpoint
        optimization_keywords: Optional[list[str]] = None,
        callbacks: Optional[list[BaseCallback]] = None
    ):
        super().__init__(callbacks)
        self.api_url = api_url or os.getenv("CUSTOM_API_URL", "http://localhost:4242/generativeai/classification/document/auto_opt")
        self.instruction_url = instruction_url or "http://localhost:4243/generativeai/classification/document/generate_instructions"
        self.optimization_keywords = optimization_keywords or os.getenv("OPTIMIZATION_KEYWORDS", "summarize,observe,optimize,propose,instruction,bootstrap,program_description,module_description").split(",")
        logger.info(f"Initialized CustomAPIAdapter with classification endpoint: {self.api_url}")
        logger.info(f"Instruction generation endpoint: {self.instruction_url}")
    
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
        formatted_inputs = self.format(signature, demos, inputs)
        is_optimization = self._is_optimization_call(inputs, formatted_inputs)
        document_text = self._extract_document_text(inputs, formatted_inputs)
        definition = getattr(signature, 'instructions', "Classification task")

        if is_optimization:
            optimization_type = self._identify_optimization_type(inputs, formatted_inputs)
            return self._generate_optimization_response(optimization_type, document_text)

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

        logger.info(f"Sending request to custom endpoint: {self.api_url}")
        logger.info(f"Document length: {len(document_text)} chars")

        try:
            response = requests.post(self.api_url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            logger.info(f"Response received: {result}")

            label = result.get("label", "unknown")
            output = {"label": label, "logprobs": None}
            output.update({k: v for k, v in result.items() if k != "label"})
            return [output]
        except Exception as e:
            logger.error(f"Error calling custom API endpoint: {e}")
            raise e

    def _is_optimization_call(self, inputs: dict[str, Any], formatted_inputs: list[dict[str, Any]]) -> bool:
        """
        Check if this call is part of the optimization process rather than a regular classification
        """
        for value in inputs.values():
            if any(keyword in str(value).lower() for keyword in self.optimization_keywords):
                return True

        for msg in formatted_inputs:
            if 'content' in msg and any(keyword in str(msg['content']).lower() for keyword in self.optimization_keywords):
                return True

        return False

    def _extract_document_text(self, inputs: dict[str, Any], formatted_inputs: list[dict[str, Any]]) -> str:
        """
        Extract document text from inputs or formatted inputs.
        """
        for field_name in ['document', 'text', 'context', 'content']:
            if field_name in inputs and isinstance(inputs[field_name], str):
                return inputs[field_name]

        for msg in formatted_inputs:
            if msg.get('role') == 'user' and 'content' in msg:
                return msg['content']

        return ""

    def _identify_optimization_type(self, inputs: dict[str, Any], formatted_inputs: list[dict[str, Any]]) -> str:
        """
        Identify what type of optimization call this is
        """
        check_texts = [str(value).lower() for value in inputs.values()]
        check_texts.extend(str(msg['content']).lower() for msg in formatted_inputs if 'content' in msg)

        if any("propose" in text and "instruction" in text for text in check_texts):
            return "propose_instruction"
        if any("observation" in text for text in check_texts):
            return "observations"
        if any("summarize" in text for text in check_texts):
            return "summary"
        if any("program_description" in text for text in check_texts):
            return "program_description"
        if any("module_description" in text for text in check_texts):
            return "module_description"

        return "generic_optimization"

    def _call_instruction_api(self, context_text: str) -> list[dict[str, Any]]:
        """
        Call the instruction generation API endpoint.
        """
        try:
            # Build the payload to match the API's expected format
            payload = {
                "task": "document_classification",
                "context": context_text[:2000],  # Limit context to first 2000 chars
                "keywords": ["energy", "market", "strategy", "policy", "regulation", "pricing", "competition"],
                "basic_instruction": "All documents or communications that discuss energy market strategy",
                "num_instructions": 5,  # Request multiple instruction candidates
                "examples": [
                    {
                        "document": """
                        Date: Wed, 15 Mar 2000 09:23:00 -0800 (PST)
                        From: John Smith
                        To: Jane Doe
                        Subject: Quarterly Energy Market Strategy

                        Hello Jane,

                        I wanted to discuss our energy market strategy for the next quarter. We should focus on:

                        1. Increasing our natural gas positions
                        2. Reducing exposure to coal markets
                        3. Developing new renewable energy partnerships

                        Let's schedule a meeting to review these strategies in detail.

                        Regards,
                        John
                        """,
                        "label": "relevant"
                    },
                    {
                        "document": """
                        Date: Wed, 15 Mar 2000 09:23:00 -0800 (PST)
                        From: Alice Johnson
                        To: Bob Williams
                        Subject: Office Supplies Order

                        Hi Bob,

                        Could you please order the following office supplies for our department:

                        - Printer paper (10 reams)
                        - Staples (5 boxes)
                        - Sticky notes (assorted colors)

                        Thanks,
                        Alice
                        """,
                        "label": "non-relevant"
                    },
                    {
                        "document": """
                        Date: Thu, 16 Mar 2000 14:45:00 -0800 (PST)
                        From: Sarah Chen
                        To: Michael Rodriguez
                        Subject: Re: New Energy Regulations Impact

                        Michael,

                        I've analyzed the new regulations and their potential impact on our market position. We should consider the following adjustments to our strategy:

                        1. Accelerate investments in renewables to meet the new requirements
                        2. Adjust our pricing models to account for the carbon tax implications
                        3. Engage with the regulatory bodies to provide input on implementation timeline

                        I've attached a detailed analysis for your review.

                        Best,
                        Sarah
                        """,
                        "label": "relevant"
                    }
                ],
                "llm_config": {},
                "request_tracker": {"request_tracking": "DSPY_INSTRUCTION_GENERATION"}
            }

            # Log the payload for debugging
            logger.info(f"Payload for instruction API: {json.dumps(payload, indent=2)}")

            # Send the request to the API
            response = requests.post(self.instruction_url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()

            # Parse the response
            result = response.json()
            logger.info(f"Instruction API response received: {result}")

            # Validate and return the instructions
            if "instructions" in result and isinstance(result["instructions"], list) and result["instructions"]:
                return [{"proposed_instruction": instruction} for instruction in result["instructions"]]
            else:
                logger.warning("Instruction API didn't return valid instructions, using fallback")
                return [{"proposed_instruction": "All documents or communications that discuss energy market strategy"}]

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error when calling instruction API: {e}")
        except ValueError as e:
            logger.error(f"Error parsing JSON response from instruction API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error when calling instruction API: {e}")

        # Fallback to a default instruction in case of errors
        return [{"proposed_instruction": "All documents or communications that discuss energy market strategy"}]

    def _generate_optimization_response(self, optimization_type: str, document_text: str) -> list[dict[str, Any]]:
        """
        Generate a response for optimization-related calls, either synthetic or via API.
        """
        if optimization_type == "propose_instruction" and self.instruction_url:
            return self._call_instruction_api(document_text)

        # Existing synthetic response logic
        response = {
            "label": "non-relevant",  # Default label
            "logprobs": None
        }

        if optimization_type == "propose_instruction":
            response["proposed_instruction"] = "All documents or communications that discuss energy market strategy"

        return [response]

    def _identify_expected_keys(self, inputs: dict[str, Any], formatted_inputs: list[dict[str, Any]]) -> list[str]:
        """
        Try to identify what keys are expected in the response based on the inputs and formatted context
        """
        expected_keys = ["label", "text", "logprobs"]

        if any("summarize" in str(x).lower() for x in inputs.values()):
            expected_keys.append("summary")

        if any("optimize" in str(x).lower() for x in inputs.values()) or any("instruction" in str(x).lower() for x in inputs.values()):
            expected_keys.append("proposed_instruction")

        if any("program" in str(x).lower() for x in inputs.values()):
            expected_keys.append("program_description")

        if any("module" in str(x).lower() for x in inputs.values()):
            expected_keys.append("module_description")

        if any("observe" in str(x).lower() for x in inputs.values()):
            expected_keys.append("observations")

        for msg in formatted_inputs:
            if 'content' in msg:
                content = str(msg['content']).lower()
                if "summarize" in content:
                    expected_keys.append("summary")
                if "optimize" in content or "instruction" in content:
                    expected_keys.append("proposed_instruction")
                if "program" in content:
                    expected_keys.append("program_description")
                if "module" in content:
                    expected_keys.append("module_description")
                if "observe" in content:
                    expected_keys.append("observations")

        return expected_keys

    def _call_post_process(self, outputs: list[dict[str, Any]], signature: Type[Signature]) -> list[dict[str, Any]]:
        values = []

        for output in outputs:
            output_logprobs = None

            if isinstance(output, dict):
                if "text" in output:
                    output_text = output["text"]
                else:
                    if "label" in output:
                        output_text = f"label: {output['label']}"
                    else:
                        output_text = str(output)

                output_logprobs = output.get("logprobs")
                other_keys = {k: v for k, v in output.items() 
                             if k not in ["text", "logprobs"]}
            else:
                output_text = str(output)
                other_keys = {}

            value = self.parse(signature, output_text)
            value["logprobs"] = output_logprobs

            for k, v in other_keys.items():
                if k != "label":
                    value[k] = v

            values.append(value)

        return values