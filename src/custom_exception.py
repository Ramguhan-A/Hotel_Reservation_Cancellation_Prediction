import traceback
import sys 
    
class CustomException(Exception):
    def __init__(self, error_message, original_error_message=None):
        super().__init__(error_message)
        self.detailed_error_message = self.get_detailed_error_message(error_message, original_error_message)
        
    @staticmethod
    def get_detailed_error_message(error_message, original_error_message):

        exc_type, exc_obj, exc_tb = sys.exc_info()  # We only need traceback
            
        if exc_tb:
            file_name = exc_tb.tb_frame.f_code.co_filename  # get frame information from traceback
            line_no = exc_tb.tb_lineno
            
            original_msg = f", {original_error_message}" if original_error_message else ""   # to get rid of None (in case)
            return f"Error occurred in file '{file_name}', line {line_no}: {error_message}, {original_error_message}"
        return error_message
    
    def __str__(self):
        return self.detailed_error_message