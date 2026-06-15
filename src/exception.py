import sys  # Importing sys to access exception info (traceback details)

def error_message_details(error, error_detail: sys):
    """
    Extracts detailed information about an exception:
    - File name where the error occurred
    - Line number of the error
    - Actual error message
    """
    
    # exc_info() returns (exception_type, exception_value, traceback_object)
    _, _, exc_tb = error_detail.exc_info()
    
    # Extract the file name from the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Extract the exact line number where the error occurred
    line_number = exc_tb.tb_lineno
    
    # Create a formatted error message
    error_message = (
        f"Error occurred in script: {file_name} "
        f"at line number: {line_number} "
        f"with error message: {str(error)}"
    )
    
    return error_message  # Return the formatted message


class CustomException(Exception):
    """
    Custom exception class that extends Python's built-in Exception.
    It automatically formats the error message using error_message_details().
    """
    
    def __init__(self, error_message, error_detail: sys):
        # Call the parent Exception class with the original error message
        super().__init__(error_message)
        
        # Create a detailed error message using our helper function
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        # When the exception is printed, return the formatted message
        return self.error_message
