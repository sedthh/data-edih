import os
import json
import time
import warnings

import numpy as np
import openai

from typing import Optional


class EDIH_WARNING(Warning):
    
    def __init__(self, message):
        self.message = message
    
    def __str__(self):
        return repr(self.message)


class EDIH_CORE:
    VERSION = "0.0.2"
    
    def __init__(self, 
                 *, 
                 user: Optional[str] = None,
                 system: str = "system.json", 
                 **kwargs) -> None:
        self._system = self._validate_system_data(system)
        self._setup_openai_api()
        self._user = str(user) if user is not None else ""
    
    def _validate_system_data(self, file: str) -> dict:
        if not os.path.exists(file):
            raise FileNotFoundError(f"System file '{file}' does not exist!")
        with open(file, "r") as f:
            try:
                data = json.loads(f.read())
            except json.decoder.JSONDecodeError as e:
                raise ValueError(f"The file '{file}' is malformed. {e}")
        if "openai" not in data.keys():
            raise ValueError(f"Nincs megadva OpenAI API kulcs a '{file}' fájlban {{'openai': '...'}} formátummal. Az alábbi címen generálhatsz sajátot: https://platform.openai.com/account/api-keys")
        elif data["openai"] == "OPEN_AI_SECRET_KEY":
            raise ValueError(f"Elfelejtettél beállítani OpenAI API kulcsot a '{file}' fájlban. Saját kulcsot itt generálhatsz: https://platform.openai.com/account/api-keys")
        if "rate_limit_per_minute" not in data:
            data["rate_limit_per_minute"] = 20
        if "website" not in data:
            data["website"] = "https://www.inf.elte.hu/"
        if "user_agent" not in data:
            data["user_agent"] = f"Mozilla/5.0 (compatible; EDIH/{self.VERSION}; +{data['website']})"
        return data
    
    def _setup_openai_api(self):
        openai.api_key = self._system["openai"]
    
    def _rate_limit_alert(self):
        warnings.warn("Túl gyorsan küldöd az üzeneteket, az OpenAI ideiglenesen lekorlátozhatja a kéréseket!", EDIH_WARNING)
       
    def retry_with_exponential_backoff(
        self,
        func,
        max_retries: int = 3,
        errors: tuple = (openai.error.RateLimitError,),
    ):
        """Retry a function with exponential backoff based on https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb"""
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = 60. / self._system["rate_limit_per_minute"]

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)

                # Retry on specified errors
                except errors as e:
                    # Increment retries
                    num_retries += 1
                    self._rate_limit_alert()
                    
                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise ValueError(f"Nem sikerült elküldeni az üzenetet {max_retries} újrapróbálkozás után sem. Valószínűleg az OpenAI ideiglenesen letiltott túlzott használat miatt.")

                    # Increment the delay
                    delay *= 2. * (1 + np.random.random())

                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper
