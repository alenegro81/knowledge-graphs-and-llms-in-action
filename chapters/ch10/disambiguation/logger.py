import logging
import os

class Logger:
    """
    A custom logger class to handle logging in a standardized way.
    """
    
    def __init__(self, log_name, log_file=None, log_level=logging.DEBUG):
        """
        Initializes the Logger instance.
        
        Args:
            log_file (str): Optional path to a log file. If not provided, logs will only go to the console.
            log_level (int): Logging level. Defaults to DEBUG.
        """
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(log_level)
        
        # Create a log formatter
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        
        # Create a stream handler for console output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(log_formatter)
        
        # Add the console handler to the logger
        self.logger.addHandler(console_handler)
        
        # Optionally log to a file
        if log_file:
            # Ensure the log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Create a file handler for logging to a file
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(log_formatter)
            
            # Add the file handler to the logger
            self.logger.addHandler(file_handler)

    def debug(self, message):
        """Logs a debug message."""
        self.logger.debug(message)

    def info(self, message):
        """Logs an info message."""
        self.logger.info(message)

    def warning(self, message):
        """Logs a warning message."""
        self.logger.warning(message)

    def error(self, message):
        """Logs an error message."""
        self.logger.error(message)

    def critical(self, message):
        """Logs a critical message."""
        self.logger.critical(message)
