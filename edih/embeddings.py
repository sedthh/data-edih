from .core import EDIH_CORE

import requests
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from typing import Optional, Union, Any


class Vector:
    
    def __init__(self,
                 text: str,
                 value: np.ndarray,
                 norm: Optional[np.ndarray] = None) -> None:
        self.text = str(text)
        self.value = np.array(value)
        self.altered = False
        if norm is None:
            self._normalize()
        else:
            self.norm = norm
        
    def _normalize(self):
        self.norm = self.value / np.linalg.norm(self.value)
    
    def __add__(self, other):
        original = deepcopy(self)
        original += other
        return original
    
    def __iadd__(self, other):
        if not isinstance(other, Vector):
            raise ValueError("Csak más embedding vektorokkal végezhetők műveletek.")
        self.text = f'{self.text} + {other.text}'
        self.value += other.value
        self.altered = True
        self._normalize()
        return self
    
    def __sub__(self, other):
        original = deepcopy(self)
        original -= other
        return original
    
    def __isub__(self, other):
        if not isinstance(other, Vector):
            raise ValueError("Csak más embedding vektorokkal végezhetők műveletek.")
        self.text = f'{self.text} - {other.text}'
        self.value -= other.value
        self.altered = True
        self._normalize()
        return self
    
    def __eq__(self, other):
        return np.allclose(self.value, other.value)
            
    def __str__(self):
        return f"* {self.text}" if self.altered else self.text
    
    def __repr__(self):
        return str(self.value)


class Embeddings(EDIH_CORE):
    
    def __init__(self, 
                 data: Optional[Union[dict, pd.DataFrame]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if data is None:
            self.data = {}
        elif isinstance(data, dict):
            self.data = deepcopy(data)
        elif isinstance(data, pd.DataFrame):
            self.data = {}
            for _, row in data.iterrows():
                self.data[row["text"]] = Vector(text=row["text"], 
                                                value=row["value"], 
                                                norm=row["norm"] if "norm" in row else None)
        else:
            raise ValueError(f"Ismeretlen adattípus: {type(data)}")
        self.total_tokens = 0
        self.total_api_calls = 0
        self.total_response_time = 0
        
    def _validate_system_data(self, file: str) -> dict:
        data = super()._validate_system_data(file)
        if "model_embedding" not in data.keys():
            data["model_embedding"] = "text-embedding-ada-002"  # version 2
        return data
    
    def __iter__(self):
        for text in self.data:
            yield text, self.data[text].value, self.data[text].norm, self.data[text].altered
    
    def _find(self, query: Any) -> dict:
        results = {}
        if isinstance(query, Vector):
            for key, value in self.data.items():
                results[key] = self.distance(query, value)
        else:
            self.embedding(query)
            for key, value in self.data.items():
                if key == query:
                    continue
                results[key] = self.distance(self.data[query], value)
        return results
    
    def find(self, query: Any, top: int = 10) -> list:
        results = self._find(query)
        return [result[0] for result in sorted(results.items(), key=lambda item: item[1])[:top]]
    
    def embedding(self, 
                  content: str, 
                 **kwargs) -> dict:
        if content in self.data:
            return self.data[content]
        
        now = time.time()
        self.total_api_calls += 1
        data = requests.post("https://api.openai.com/v1/embeddings", 
                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._system['openai']}"}, 
                             json={"input": str(content), "model": self._system['model_embedding']})
        data.raise_for_status()
        data = data.json()
        if "error" in data.keys():  # data["error"]["type"]
            raise ValueError(data["error"]["message"])
        
        self.total_tokens += data["usage"]["total_tokens"]
        self.total_response_time += time.time() - now
        
        embedding = np.array(data["data"][0]["embedding"])
        self.data[content] = Vector(str(content), embedding)
        return self.data[content]
    
    def distance(self, a: Any, b: Any):
        """0. (megegyező) és 2. (ellentétes) között ad vissza értéket"""
        if isinstance(a, str) and (b, str):
            a, b = self.embedding(a), self.embedding(b)
        if isinstance(a, Vector) and isinstance(b, Vector):
            return 1. - np.dot(a.norm, b.norm)
        return cosine(a, b)
    
    def similarity(self, a: Any, b: Any):
        """1. (megegyező) és -1. (ellentétes) között ad vissza értéket"""
        return 1. - self.distance(a, b)

    def to_pandas(self):
        df = []
        for text, value, norm, altered in self:
            if not altered:
                df.append({"text": text, "value": value, "norm": norm})
        return pd.DataFrame(df)
        