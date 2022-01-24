import inspect
from tensorflow.keras import losses, optimizers, activations
from tensorflow.keras import backend as K


def get_losses():
    """introspect keras API and get losses"""
    loss_func_map = {}
    for name, obj in inspect.getmembers(losses):
        if inspect.isfunction(obj):
            args = list(inspect.signature(obj).parameters.keys())
            # FIXME: do it in a cleaner way
            if len(args) == 2 and args[0] == "y_true" and args[1] == "y_pred":
                loss_func_map[obj] = name
    return {v: k for k, v in loss_func_map.items()}


def get_optimizers():
    """introspect keras API and get optimizers"""
    optimizer_map = {}
    for name, obj in inspect.getmembers(optimizers):
        if inspect.isclass(obj):
            try:
                args = list(inspect.signature(obj).parameters.keys())
                # FIXME: do it in a cleaner way
                if "learning_rate" in args:
                    optimizer_map[obj] = name
            except Exception:
                pass

    return {v: k for k, v in optimizer_map.items()}


def get_activations():
    """introspect keras API and get activations"""
    activation_map = {}
    for name, obj in inspect.getmembers(activations):
        if inspect.isfunction(obj):
            try:
                args = list(inspect.signature(obj).parameters.keys())
                # FIXME: do it in a cleaner way
                if "x" in args:
                    activation_map[obj] = name
            except Exception:
                pass

    return {v: k for k, v in activation_map.items()}


def r_square(y_true, y_pred):
    """custom metric to compute rsquare"""
    res_ss = K.sum(K.square(y_true - y_pred))
    tot_ss = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - res_ss / (tot_ss + K.epsilon())
