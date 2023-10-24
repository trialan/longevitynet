
import time

def retry_if_exception(wait_time=1):
    """Decorator that retries the function upon an exception."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error encountered: {e}. Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
        return wrapper
    return decorator
