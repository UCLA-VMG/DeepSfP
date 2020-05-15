from types import ModuleType
from typing import Any, Dict, List, Union, Callable
from functools import partial


class ComponentFactory:
    def __init__(self, component_type: str = None, components: Dict[str, Any] = {}, 
                modules: List[ModuleType] = [], partial: bool = False, default: str = '',
                allow_none: bool = False):
        self._component_type = component_type
        self._components = components
        self._modules = modules
        self._partial = partial
        self._default = default
        self._allow_none = allow_none

    def _register(self, component: Any):
        name = component.__name__
        assert name not in self._components, f'{self._component_type} "{name}" exists!'
        self._components[name] = component

    def register(self, component: Any = None):
        if not component:  # If called as decorator...
            def wrapper(_component: Any):
                self._register(_component)
                return _component
            return wrapper

        self._register(component)

    def _build(self, component, **kwargs) -> Union[Callable, "partial[Callable]"]:
        if self._partial:
            return partial(component, **kwargs)
        return component(**kwargs)

    def build(self, name: str = '', **kwargs):
        if not name:
            name = self._default
        # Check registered components
        if name in self._components:
            return self._build(self._components[name], **kwargs)
        # Check packages
        for m in self._modules:
            if hasattr(m, name):
                return self._build(getattr(m, name), **kwargs)
        if self._allow_none:
            return None
        raise NotImplementedError(f'{name} {self._component_type} does not exist.'
                    f' Must choose from: {self._components.keys()}')
