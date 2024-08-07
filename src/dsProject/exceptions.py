import sys
from src.dsProject.logger import logging


class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message, error_details)

    def __str__(self):
        return self.error_message
    

def error_message_detail(error, error_details:sys):
        _,_,exc_tb=error_details.exc_info()
        filename = exc_tb.tb_frame.f_code.co_filename
        error_messsage="Error occured in python script name [{0}] : line number [{1}] : error message : [{2}]" .format(filename, exc_tb.tb_lineno, str(error))

        return error_messsage