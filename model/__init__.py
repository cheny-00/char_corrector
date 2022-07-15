
def model_table(model_name):
    from .rnnlm import RNNLM
    
    models = {
        "rnnlm": RNNLM,
        
    }
    return models[model_name]