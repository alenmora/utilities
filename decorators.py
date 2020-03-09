def applyToItems(func: Function) -> Wrapper:
"""
Applies the function func to all the values in the dictionary, and modifies the dictionary in place.
If the dictionary is composed of dictionaries, it recursively goes down until the first value which
is not a dictionary
"""
    def wrapper(*args,**kwargs):
        if isinstance(args[0], dict):
            for key,val in args[0].items(): args[0][key]=wrapper(val, *args[1:],**kwargs)
            return args[0]
        
        else: return func(*args,**kwargs)
    return wrapper
