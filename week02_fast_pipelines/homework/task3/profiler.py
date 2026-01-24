import json
import time
import torch
import os
from collections import defaultdict


class Profile:
    def __init__(self, model, name="model", schedule=None):
        self.name_map = self._build_name_map(model, name)
        self.events = []
        
        ### TODO
    
    def _build_name_map(self, model, name="model"):
        name_map = {}
        for full_name, module in model.named_modules():
            if full_name == "":
                full_name = name

            if self._is_leaf(module):
                name_map[module] = module.__class__.__name__
            else:
                name_map[module] = f"{full_name}: {module.__class__.__name__}"

        return name_map

    def _is_leaf(self, module):
        return len(list(module.children())) == 0

    def _forward_pre_hook(self, module, inputs):
        ### TODO
        raise NotImplementedError

    def _forward_post_hook(self, module, inputs, outputs):
        ### TODO
        raise NotImplementedError

    def _backward_pre_hook(self, module, grad_output):
        ### TODO
        raise NotImplementedError

    def _backward_post_hook(self, module, grad_input, grad_output):
        ### TODO
        raise NotImplementedError

    def __enter__(self):
        ### TODO
        raise NotImplementedError
 
    def __exit__(self, type, value, traceback):
        ### TODO
        raise NotImplementedError

    def step(self):
        ### TODO
        raise NotImplementedError

    def summary(self):
        print("Summary:")
        for event in self.events:
            print(event)

    def to_perfetto(self, path="trace.json"):
        ### TODO
        raise NotImplementedError
