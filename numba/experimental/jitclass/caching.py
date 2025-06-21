"""
JitClass caching support for Numba
"""

import hashlib
import pickle
from collections import OrderedDict

from numba.core import types
from numba.core.caching import Cache
from numba.core.serialize import _rebuild_reduction


class JitClassCache(Cache):
    """
    Cache for compiled jitclass definitions.
    
    Unlike function caching, jitclass caching operates at the class level,
    caching the entire compiled class including all methods, type definitions,
    and boxing infrastructure.
    """
    
    def __init__(self, pyfunc, cache_dir=None):
        # For jitclass, pyfunc is the class being decorated  
        self._pyfunc = pyfunc
        super().__init__(cache_dir)
        
    def _get_cache_key(self, spec, methods, properties, static_methods):
        """
        Generate a deterministic cache key for a jitclass.
        
        The key includes:
        - Class name and module
        - Field specification (names and types)
        - Method bytecode hashes
        - Property and static method definitions
        """
        cls = self._pyfunc
        
        # Hash the class specification
        spec_data = []
        if hasattr(spec, 'items'):
            spec_data = list(spec.items())
        else:
            spec_data = list(spec)
        
        # Sort for deterministic ordering
        spec_data = sorted(spec_data, key=lambda x: x[0])
        
        # Hash method bytecode
        method_hashes = []
        for name, method in sorted(methods.items()):
            if hasattr(method, '__code__'):
                method_hashes.append((name, method.__code__.co_code))
            else:
                # For pre-compiled methods, use string representation
                method_hashes.append((name, str(method)))
        
        # Hash properties
        prop_hashes = []
        for name, prop in sorted(properties.items()):
            if hasattr(prop, 'fget') and hasattr(prop.fget, '__code__'):
                prop_hashes.append((name, 'get', prop.fget.__code__.co_code))
            if hasattr(prop, 'fset') and hasattr(prop.fset, '__code__'):
                prop_hashes.append((name, 'set', prop.fset.__code__.co_code))
        
        # Hash static methods
        static_hashes = []
        for name, method in sorted(static_methods.items()):
            if hasattr(method, '__code__'):
                static_hashes.append((name, method.__code__.co_code))
        
        # Create composite key
        key_components = (
            cls.__name__,
            cls.__module__,
            spec_data,
            method_hashes,
            prop_hashes,
            static_hashes,
        )
        
        # Generate stable hash
        key_bytes = pickle.dumps(key_components, protocol=pickle.HIGHEST_PROTOCOL)
        key_hash = hashlib.sha256(key_bytes).hexdigest()
        
        return key_hash
    
    def _get_index_key(self, spec, methods, properties, static_methods):
        """Generate index key for the class cache"""
        cache_key = self._get_cache_key(spec, methods, properties, static_methods)
        
        # Include additional metadata for index
        cls = self._pyfunc
        return (
            cache_key,
            cls.__name__,
            cls.__module__ if cls.__module__ else '<unknown>',
        )
    
    def _get_disambiguator(self):
        """Get disambiguator for cache file naming"""
        cls = self._pyfunc
        return f"{cls.__name__}-{hash(cls.__module__ or '') & 0xffffffff:x}"
    
    def save_class(self, spec, methods, properties, static_methods, 
                   class_type, method_compile_results, boxing_data):
        """
        Save compiled jitclass to cache.
        
        Parameters:
        - spec: Field specification 
        - methods: Dict of method name -> method function
        - properties: Dict of property name -> property object
        - static_methods: Dict of static method name -> method function
        - class_type: Compiled ClassType instance
        - method_compile_results: Dict of method compile results
        - boxing_data: Boxing infrastructure data
        """
        index_key = self._get_index_key(spec, methods, properties, static_methods)
        
        # Serialize class data
        class_data = JitClassCacheData(
            spec=spec,
            methods=methods,
            properties=properties, 
            static_methods=static_methods,
            class_type=class_type,
            method_compile_results=method_compile_results,
            boxing_data=boxing_data,
        )
        
        # Save to cache
        self.save(index_key, class_data)
        
    def load_class(self, spec, methods, properties, static_methods):
        """
        Load compiled jitclass from cache.
        
        Returns cached class data if found, None otherwise.
        """
        index_key = self._get_index_key(spec, methods, properties, static_methods)
        
        try:
            return self.load(index_key)
        except (KeyError, FileNotFoundError, EOFError):
            return None


class JitClassCacheData:
    """
    Container for cached jitclass compilation data.
    
    This class handles serialization of all components needed to 
    reconstruct a compiled jitclass.
    """
    
    def __init__(self, spec, methods, properties, static_methods,
                 class_type, method_compile_results, boxing_data):
        self.spec = spec
        self.methods = methods
        self.properties = properties
        self.static_methods = static_methods
        self.class_type = class_type
        self.method_compile_results = method_compile_results
        self.boxing_data = boxing_data
    
    def __reduce__(self):
        """Support for pickle serialization"""
        return (_rebuild_jitclass_cache_data, (
            self.spec,
            self.methods,
            self.properties,
            self.static_methods,
            self.class_type,
            self.method_compile_results,
            self.boxing_data,
        ))


def _rebuild_jitclass_cache_data(spec, methods, properties, static_methods,
                                class_type, method_compile_results, boxing_data):
    """Rebuild JitClassCacheData from pickle data"""
    return JitClassCacheData(
        spec, methods, properties, static_methods,
        class_type, method_compile_results, boxing_data
    )


def make_deterministic_class_id(cls_name, spec, methods_hash):
    """
    Create deterministic class ID to replace runtime id() calls.
    
    This replaces the non-deterministic id(self) used in class type names
    with a stable hash based on class definition.
    """
    content = f"{cls_name}:{spec}:{methods_hash}"
    return hash(content) & 0xffffffff


def compute_class_hash(cls, spec, methods):
    """
    Compute stable hash of complete class definition.
    
    Used for deterministic type IDs and cache keys.
    """
    spec_items = []
    if hasattr(spec, 'items'):
        spec_items = sorted(spec.items())
    else:
        spec_items = sorted(list(spec))
    
    method_codes = {}
    for name, method in methods.items():
        if hasattr(method, '__code__'):
            method_codes[name] = method.__code__.co_code
        else:
            method_codes[name] = str(method).encode('utf-8')
    
    components = (
        cls.__name__,
        cls.__module__ or '<unknown>',
        tuple(spec_items),
        tuple(sorted(method_codes.items())),
    )
    
    content_bytes = pickle.dumps(components, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(content_bytes).hexdigest()