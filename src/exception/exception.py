

class Error(Exception):
    '''A base clase for exception in this module'''
    pass


class InputError(Error):
    '''Exception raised for errors in input.
    
    Attributes:
    
    '''

    def __init_(self, expression, message):
        self.expression = expression
        self.message = message
