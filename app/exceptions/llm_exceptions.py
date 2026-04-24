# app/exceptions/llm_exceptions.py

class LLMSchemaValidationError(Exception):
    """
    An exception thrown when the output from LLM does not conform to the JSON format or schema rules specified the prompt.
    """
    def __init__(self, message, raw_llm_output=None, errors=None):
        super().__init__(message)
        self.message = message
        self.raw_llm_output = raw_llm_output  # Optional: for debugging
        self.errors = errors  # List detail error validation

    def __str__(self):
        return f"LLMSchemaValidationError: {self.message} | Total Errors: {len(self.errors) if self.errors else 0}"