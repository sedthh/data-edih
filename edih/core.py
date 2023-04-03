import os
import re
import json
import requests
from datetime import datetime
import time
import warnings
from copy import deepcopy

try: 
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import tabulate, openpyxl

import openai

from typing import Optional, Any


class EDIH_GPT_WARNING(Warning):
    
    def __init__(self, message):
        self.message = message
    
    def __str__(self):
        return repr(self.message)


class EDIH_GPT:
    
    VERSION = "0.0.1"
    COMMANDS = ["DATE", "FILE", "URL", "CSV", "EXCEL", "JSON"]
    
    def __init__(self, 
                 context: Optional[str] = None,
                 *, 
                 user: Optional[str] = None,
                 system: str = "system.json", 
                 **kwargs) -> None:
        self._system = self._validate_system_data(system)
        self._setup_openai_api()
        
        self._user = str(user) if user is not None else ""
        self._context = context
        self.reset()
        
    
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
        if "model" not in data.keys():
            data["model"] = "gpt-3.5-turbo"  # gpt-4-32k
        if "website" not in data:
            data["website"] = "https://www.inf.elte.hu/"
        if "user_agent" not in data:
            data["user_agent"] = f"Mozilla/5.0 (compatible; EDIH_GPT/{self.VERSION}; +{data['website']})"
        if "parse" not in data:
            data["parse"] = deepcopy(self.COMMANDS)
        elif not data["parse"]:
            data["parse"] = []
        return data
    
    def _setup_openai_api(self):
        openai.api_key = self._system["openai"]
    
    def reset(self):
        self.history = []
        self.tokens = {
            "last_sent": 0,
            "all_sent": 0,
            "last_received": 0,
            "all_received": 0,
            "last_sum": 0,
            "all_sum": 0,
        }
        self.total_api_calls = 0
        self.total_response_time = 0.
        if self._context is not None:
            self.messages = [{"role": "system", 
                              "content": self._parse_context(self._context)}]
        else:
            self.messages = []
    
    @property
    def context(self):
        if self._context is not None:
            return self.messages[0]["content"]
        else:
            return ""
    
    def _parse_context(self, text: str) -> str:
        if self._system["parse"]:
            return re.sub(r'\{([A-Z]+)\:\:([^\}]+?)\}', self._magic_words, text)
        else:
            return text
    
    def _magic_words(self, match) -> str:
        if str(match.group(1)) not in self._system["parse"]:
            return f"{{{match.group(1)}::{match.group(2)}}}"
        if match.group(1) == "DATE":
            text = datetime.now().strftime(match.group(2))
            for eng, hun in zip(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                ['Hétfő', 'Kedd', 'Szerda', 'Csütörtök', 'Péntek', 'Szombat', 'Vasárnap']):
                text = text.replace(eng, hun)
            return text
        elif match.group(1) == "FILE":
            with open(str(match.group(2)), "r", encoding="utf-8") as f:
                return f.read()
        elif match.group(1) == "URL":
            parts = str(match.group(2)).split("::")
            html = self._request(parts[0]).strip()
            html = html.replace("<br>", "<br/>")
            dom = BeautifulSoup(html, "html.parser")
            if dom.find("body") is not None:
                dom = dom.find("body")
                for s in dom.select("script"):
                    s.extract()
            for i, part in enumerate(parts[1:]):
                if "#" in part:
                    if part.startswith("#"):
                        dom = dom.find("div", {"id": part[1:]})
                    else:
                        dom = dom.find(part.split("#")[0], {"id": part.split("#")[1]})
                elif "." in part:
                    if part.startswith("."):
                        dom = dom.find("div", {"class": part[1:]})
                    else:
                        dom = dom.find(part.split(".")[0], {"class": part.split(".")[1]})
                else:
                    dom = dom.find(part)
                if dom is None:
                    raise ValueError(f"Could not find {' > '.join(parts[1:2+i])}")
            text = dom.text.replace("&amp;", "&")
            text = "\n".join([t.strip() for t in text.splitlines()])
            return re.sub(r'\n{2,}', '\n\n', text).strip()
        elif match.group(1) == "CSV":
            csv = pd.read_csv(match.group(2))
            return csv.to_markdown(index=False)
        elif match.group(1) == "EXCEL":
            parts = str(match.group(2)).split("::", 2)
            if len(parts) == 3:
                file, sheet, header = parts[0], parts[1], max(0, int(parts[2]) - 1)
            elif len(parts) == 2:
                file, sheet, header = parts[0], parts[1], None
            else:
                file, sheet, header = parts[0], 0, None
            excel = pd.read_excel(open(file, "rb"), sheet_name=sheet, header=header, engine="openpyxl")
            if header is None:
                excel.columns = [alpha for _, alpha in zip(excel.columns, [chr(value) for value in range(ord("A"), ord("A") + 26)])]
            return excel.to_markdown(index=False)
        elif match.group(1) == "JSON":
            parts = str(match.group(2)).split("::")
            payload = self._request(parts[0])
            if isinstance(payload, str):
                payload = json.loads(payload)
            for part in parts[1:]:
                payload = payload[part]
            return str(payload)
    
    def _request(self, url: str, json: bool = False) -> Optional[str]:
        data = requests.get(url, headers={"User-Agent": self._system["user_agent"]})
        data.raise_for_status()
        if json:
            return data.json()
        try:
            return data.content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return data.content.decode("ISO-8859-1")
            except UnicodeDecodeError:
                return data.content.decode("utf-8", "backslashreplace")
                
    def cost(self, key: str = "all_sum", per_token: float = 0.002 / 1000.):
        if key not in self.tokens:
            raise ValueError(f"Ismeretlen '{key}', csak az alábbi értékek érhetők el: {', '.join(list(self.tokens.keys()))}")
        return self.tokens[key] * per_token
    
    def log(self, *, file: Optional[str] = None, end: str = "\n\n") -> str:
        text = [f"{event['role'].upper()}: {event['content']}" for event in self.messages]
        text = end.join(text)
        if file is not None:
            with open(file, "w+", newline=None, encoding="utf-8") as f:
                f.write(text)    
        return text
        
    def _content_filter_alert(self):
        warnings.warn("Ez az üzenet sérti a moderálási irányelveinket!", EDIH_GPT_WARNING)
        
    def _content_length_alert(self):
        warnings.warn("Az üzenetet levágtuk, mert hosszabb a megengedettnél!", EDIH_GPT_WARNING)
    
    def chat(self, 
             content: str, 
             **kwargs) -> str:
        self.messages.append({"role": "user", 
                              "content": content})
        
        now = time.time()
        self.total_api_calls += 1
        response = openai.ChatCompletion.create(
            model = self._system["model"],
            messages = self.messages,
            n = 1,
            stream = False,
            user = self._user,
            **kwargs
        )
        self.total_response_time += time.time() - now
        
        self.history.append(response)
        
        self.tokens["last_received"] += int(response["usage"]["completion_tokens"])
        self.tokens["all_received"] += self.tokens["last_received"]
        self.tokens["last_sent"] += int(response["usage"]["total_tokens"]) - self.tokens["last_sum"] - int(response["usage"]["completion_tokens"])
        self.tokens["all_sent"] += self.tokens["last_sent"]
        self.tokens["last_sum"] = int(response["usage"]["total_tokens"])
        self.tokens["all_sum"] += self.tokens["last_sum"] 
        
        if response["choices"][0]["finish_reason"] == "content_filter":
            self._content_filter_alert()
        elif response["choices"][0]["finish_reason"] == "length":
            self._content_length_alert()
        self.messages.append({"role": response["choices"][0]["message"]["role"], 
                              "content": response["choices"][0]["message"]["content"]})
        return self.messages[-1]["content"]
