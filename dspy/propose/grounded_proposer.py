import random

import dspy
from dspy.propose.dataset_summary_generator import create_dataset_summary
from dspy.propose.utils import create_example_string, create_predictor_level_history_string, strip_prefix, get_dspy_source_code
from dspy.teleprompt.utils import get_signature, get_prompt_model
from dspy.propose.propose_base import Proposer
import requests

# Hardcoded variables (TODO: update)
MAX_INSTRUCT_IN_HISTORY = 5  # 10

TIPS = {
        "none": "",
        "creative": "Don't be afraid to be creative when creating the new instruction!",
        "simple": "Keep the instruction clear and concise.",
        "description": "Make sure your instruction is very informative and descriptive.",
        "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
        "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")',
    }

### SIGNATURES USED TO HELP WITH INSTRUCTION GENERATION ###

class DescribeProgram(dspy.Signature):
    (
        """Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe what type of task this program appears to be designed to solve, and how it appears to work."""
    )
    program_code = dspy.InputField(
        format=str,
        desc="Pseudocode for a language model program designed to solve a particular task.",
        prefix="PROGRAM CODE:",
    )
    program_example = dspy.InputField(
        format=str,
        desc="An example of the program in use.",
        prefix="EXAMPLE OF PROGRAM IN USE:",
    )
    program_description = dspy.OutputField(
        desc="Describe what task the program is designed to solve, and how it goes about solving this task.",
        prefix="SUMMARY OF PROGRAM ABOVE:",
    )


class DescribeModule(dspy.Signature):
    (
        """Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe the purpose of one of the specified module in this pipeline."""
    )
    program_code = dspy.InputField(
        format=str,
        desc="Pseudocode for a language model program designed to solve a particular task.",
        prefix="PROGRAM CODE:",
    )
    program_example = dspy.InputField(
        format=str,
        desc="An example of the program in use.",
        prefix="EXAMPLE OF PROGRAM IN USE:",
    )
    program_description = dspy.InputField(
        desc="Summary of the task the program is designed to solve, and how it goes about solving it.",
        prefix="SUMMARY OF PROGRAM ABOVE:",
    )
    module = dspy.InputField(
        desc="The module in the program that we want to describe.", prefix="MODULE:",
    )
    module_description = dspy.OutputField(
        desc="Description of the module's role in the broader program.",
        prefix="MODULE DESCRIPTION:",
    )


def generate_instruction_class(
    use_dataset_summary=True,
    program_aware=True,
    use_task_demos=True,
    use_instruct_history=True,
    use_tip=True,
):
    class GenerateSingleModuleInstruction(dspy.Signature):
        (
            """Use the information below to learn about a task that we are trying to solve using calls to an LM, then generate a new instruction that will be used to prompt a Language Model to better solve the task."""
        )
        if use_dataset_summary:
            dataset_description = dspy.InputField(
                desc="A description of the dataset that we are using.",
                prefix="DATASET SUMMARY:",
            )
        if program_aware:
            program_code = dspy.InputField(
                format=str,
                desc="Language model program designed to solve a particular task.",
                prefix="PROGRAM CODE:",
            )
            program_description = dspy.InputField(
                desc="Summary of the task the program is designed to solve, and how it goes about solving it.",
                prefix="PROGRAM DESCRIPTION:",
            )
            module = dspy.InputField(
                desc="The module to create an instruction for.", prefix="MODULE:",
            )
            module_description = dspy.InputField(
                desc="Description of the module to create an instruction for.", prefix="MODULE DESCRIPTION:",
            )
        task_demos = dspy.InputField(
            format=str,
            desc="Example inputs/outputs of our module.",
            prefix="TASK DEMO(S):",
        )
        if use_instruct_history:
            previous_instructions = dspy.InputField(
                format=str,
                desc="Previous instructions we've attempted, along with their associated scores.",
                prefix="PREVIOUS INSTRUCTIONS:",
            )
        basic_instruction = dspy.InputField(
            format=str, desc="Basic instruction.", prefix="BASIC INSTRUCTION:",
        )
        if use_tip:
            tip = dspy.InputField(
                format=str,
                desc="A suggestion for how to go about generating the new instruction.",
                prefix="TIP:",
            )
        proposed_instruction = dspy.OutputField(
            desc="Propose an instruction that will be used to prompt a Language Model to perform this task.",
            prefix="PROPOSED INSTRUCTION:",
        )

    return dspy.Predict(GenerateSingleModuleInstruction)

### CLASS RESPONSIBLE FOR GENERATING A NEW INSTRUCTION, USING THE HELPER SIGNATURES ABOVE ###

class GenerateModuleInstruction(dspy.Module):
    def __init__(
        self,
        program_code_string=None,
        use_dataset_summary=True,
        program_aware=False,
        use_task_demos=True,
        use_instruct_history=True,
        use_tip=True,
        verbose=False,
    ):
        super().__init__()
        self.use_dataset_summary = use_dataset_summary
        self.program_aware = program_aware
        self.use_task_demos = use_task_demos
        self.use_instruct_history = use_instruct_history
        self.use_tip = use_tip
        self.verbose = verbose

        self.program_code_string = program_code_string

    def forward(
        self,
        demo_candidates,
        pred_i,
        demo_set_i,
        program,
        previous_instructions,
        data_summary,
        num_demos_in_context=3,
        tip=None,
    ):
        basic_instruction = get_signature(program.predictors()[pred_i]).instructions
        task_demos = "No task demos provided."

        if self.use_task_demos and demo_candidates:
            task_demos = "\n\n".join(
                create_example_string(
                    get_signature(program.predictors()[pred_i]).fields, example
                )
                for example in demo_candidates[pred_i][demo_set_i][:num_demos_in_context]
            )

        # Ensure context is populated
        context = data_summary if data_summary else "Default context: No dataset summary available."

        # Validate examples
        examples = [
            {"document": example.document, "label": example.label}
            for example in demo_candidates[pred_i][demo_set_i][:3]
        ] if demo_candidates and demo_candidates[pred_i][demo_set_i] else []

        if not examples:
            examples = [{"document": "Default document", "label": "Default label"}]

        payload = {
            "task": "document_classification",
            "context": context,
            "keywords": ["energy", "market", "strategy", "policy", "regulation", "pricing", "competition"],
            "basic_instruction": basic_instruction,
            "num_instructions": 1,
            "examples": examples,
        }

        if tip:
            payload["tip"] = tip

        # Log the payload for debugging
        if self.verbose:
            print("API Payload:", payload)

        try:
            response = requests.post(
                "http://localhost:4243/generativeai/classification/document/generate_instructions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()

            proposed_instruction = result.get("instructions", [
                "Default instruction: All documents or communications that discuss energy market strategy."
            ])[0]
        except requests.exceptions.RequestException as e:
            print(f"Error calling instruction API: {e}")
            print("Payload that caused the error:", payload)
            proposed_instruction = "Default instruction: All documents or communications that discuss energy market strategy."

        return dspy.Prediction(proposed_instruction=proposed_instruction)

### CLASS USED TO GENERATE THE FULL SET OF INSTRUCTIONS GIVEN THE SPECIFIED CRITERIA ###

class GroundedProposer(Proposer):
    def __init__(
        self,
        prompt_model,
        program,
        trainset,
        view_data_batch_size=10,
        use_dataset_summary=True,
        program_aware=True,
        use_task_demos=True,
        num_demos_in_context = 3,
        use_instruct_history=True,
        use_tip=True,
        set_tip_randomly=True,
        set_history_randomly=True,
        verbose=False,
        rng=None
    ):
        super().__init__()
        self.program_aware = program_aware
        self.use_dataset_summary = use_dataset_summary
        self.use_task_demos = use_task_demos
        self.num_demos_in_context = num_demos_in_context
        self.use_instruct_history = use_instruct_history
        self.use_tip = use_tip
        self.set_tip_randomly=set_tip_randomly
        self.set_history_randomly=set_history_randomly
        self.verbose = verbose
        self.rng = rng or random

        self.prompt_model = get_prompt_model(prompt_model)

        self.program_code_string = None
        if self.program_aware:
            try:
                self.program_code_string = get_dspy_source_code(program)
                if self.verbose:
                    print("SOURCE CODE:",self.program_code_string)
            except Exception as e:
                print(f"Error getting source code: {e}.\n\nRunning without program aware proposer.")
                self.program_aware = False

        self.data_summary  = None
        if self.use_dataset_summary:
            try:
                self.data_summary = create_dataset_summary(
                    trainset=trainset, 
                    view_data_batch_size=view_data_batch_size, 
                    prompt_model=prompt_model,
                    api_url="http://localhost:4242/generativeai/classification/document/auto_opt"  # Add API endpoint
                )
                if self.verbose:
                    print(f"DATA SUMMARY: {self.data_summary}")
            except Exception as e:
                print(f"Error getting data summary: {e}.\n\nRunning without data aware proposer.")
                self.use_dataset_summary = False
                print("")

    def propose_instructions_for_program(
        self,
        trainset,
        program,
        demo_candidates,
        trial_logs,
        N,
        T,
    ) -> list[str]:
        """This method is responsible for returning the full set of new instructions for our program, given the specified criteria."""

        proposed_instructions = {}      

        if self.set_history_randomly:
            # Randomly select whether or not we're using instruction history
            use_history = self.rng.random() < 0.5
            self.use_instruct_history = use_history
            if self.verbose:
                print(f"Use history T/F: {self.use_instruct_history}")

        if not demo_candidates:
            if self.verbose:
                print("No demo candidates provided. Running without task demos.")
            self.use_task_demos = False
            # When no demo candidates are provided, defailt to N
            num_demos = N
        else:
            num_demos = max(len(demo_candidates[0]), 1)

        # Create an instruction for each predictor 
        for pred_i, predictor in enumerate(program.predictors()):
            for demo_set_i in range(num_demos):
                if pred_i not in proposed_instructions:
                    proposed_instructions[pred_i] = []
                selected_tip = None
                if self.set_tip_randomly:
                    if self.verbose:
                        print("Using a randomly generated configuration for our grounded proposer.")
                    # Randomly select the tip
                    selected_tip_key = self.rng.choice(list(TIPS.keys()))
                    selected_tip = TIPS[selected_tip_key]
                    self.use_tip = bool(
                        selected_tip,
                    )
                    if self.verbose:
                        print(f"Selected tip: {selected_tip_key}")

                proposed_instructions[pred_i].append(
                    self.propose_instruction_for_predictor(
                        program=program,
                        predictor=predictor,
                        pred_i=pred_i,
                        T=T,
                        demo_candidates=demo_candidates,
                        demo_set_i=demo_set_i,
                        trial_logs=trial_logs,
                        tip=selected_tip,
                    ),
                )
        
        return proposed_instructions

    def propose_instruction_for_predictor(
        self,
        program,
        predictor,
        pred_i,
        T,
        demo_candidates,
        demo_set_i,
        trial_logs,
        tip=None,
    ) -> str:
        """This method is responsible for returning a single instruction for a given predictor, using the specified criteria."""

        # Create an instruction history string for our predictor
        instruction_history = create_predictor_level_history_string(
            program, pred_i, trial_logs, MAX_INSTRUCT_IN_HISTORY,
        )

        # Create our instruction generator class (given specific criteria for this round of proposal)
        instruction_generator = GenerateModuleInstruction(
            program_code_string=self.program_code_string,
            use_dataset_summary=self.use_dataset_summary,
            program_aware=self.program_aware,
            use_task_demos=self.use_task_demos and demo_candidates,
            use_instruct_history=self.use_instruct_history and instruction_history,
            use_tip=self.use_tip,
            verbose=self.verbose
        )

        # Generate a new instruction for our predictor, using the temperature specified for this round
        original_temp = self.prompt_model.kwargs["temperature"]

        epsilon = self.rng.uniform(0.01, 0.05)
        modified_temp = T + epsilon

        with dspy.settings.context(lm=self.prompt_model):
            self.prompt_model.kwargs["temperature"] = modified_temp
            proposed_instruction = instruction_generator.forward(
                demo_candidates=demo_candidates,
                pred_i=pred_i,
                demo_set_i=demo_set_i,
                program=program,
                data_summary=self.data_summary,
                previous_instructions=instruction_history,
                num_demos_in_context = self.num_demos_in_context,
                tip=tip,
            ).proposed_instruction
        self.prompt_model.kwargs["temperature"] = original_temp

        # Log the trace used to generate the new instruction, along with the new instruction itself
        if self.verbose:
            self.prompt_model.inspect_history(n=1)
            print(f"PROPOSED INSTRUCTION: {proposed_instruction}")

        return strip_prefix(proposed_instruction)
