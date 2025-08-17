import sys

# class CustomException(Exception):
#     def __init__(self, message, original_exception=None):
#         super().__init__(message)
#         self.message = message
#         self.detailed_error_message = self.get_detailed_error_message(message)
        
#     @staticmethod
#     def get_detailed_error_message(message):

#         exc_type, exc_obj, exc_tb = sys.exc_info()  # We only need traceback
            
#         if exc_tb:
#             file_name = exc_tb.tb_frame.f_code.co_filename  # get frame information from traceback
#             line_no = exc_tb.tb_lineno
                
#             return f"Error occurred in file '{file_name}', line {line_no}: {message}"
#         return message
    
#     def __str__(self):
#         return self.detailed_error_message

import traceback

class CustomException(Exception):
    def __init__(self, message, original_error_message=None):
        super().__init__(message)
        # self.message = message
        self.detailed_error_message = self.get_detailed_error_message(message, original_error_message)
        
    @staticmethod
    def get_detailed_error_message(message, original_error_message):

        exc_type, exc_obj, exc_tb = sys.exc_info()  # We only need traceback
            
        if exc_tb:
            file_name = exc_tb.tb_frame.f_code.co_filename  # get frame information from traceback
            line_no = exc_tb.tb_lineno
            
            # original_msg = f", {original_error_message}" if original_error_message else ""
            return f"Error occurred in file '{file_name}', line {line_no}: {message}, {original_error_message}"
        return message
    
    def __str__(self):
        return self.detailed_error_message
    

if __name__ == "__main__":
    
    def divide(a, b):
        a = 4
        b = 0
        try:
            # if a>6:
            c = a / b
            # else:
            #     raise CustomException("Division by zero is not allowed")

        except CustomException as e:
            print(e)
            # raise CustomException(e)

    divide(2,0)
