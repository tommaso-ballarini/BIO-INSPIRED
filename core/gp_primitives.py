import operator
import random
from deap import gp

def protected_div(left, right):
    # Se il denominatore è quasi zero, restituisci 1 invece di rischiare errori
    if abs(right) < 1e-6:
        return 1
    return left / right
    
def if_then_else(input, output1, output2):
    return output1 if input else output2

def random_101():
    return random.randint(-1, 1)

def setup_primitives(n_features=11):
    """
    Configura il set di primitive per DEAP.
    n_features: Dimensione dell'input (11 per Freeway wrapper)
    """
    # Input set: ARG0, ARG1, ... ARG10
    pset = gp.PrimitiveSet("MAIN", n_features)
    
    # Operatori matematici
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    
    # Operatori logici/condizionali (utili per decisioni)
    pset.addPrimitive(max, 2)
    pset.addPrimitive(min, 2)
    pset.addPrimitive(operator.lt, 2) # Less than
    pset.addPrimitive(operator.gt, 2) # Greater than
    pset.addPrimitive(if_then_else, 3)

    # Costanti effimere (numeri casuali che il GP può usare come soglie)
    pset.addEphemeralConstant("rand101", random_101)    
    # Rinomina gli argomenti per leggibilità (opzionale ma utile per il debug)
    # Basato sul tuo FreewayOCAtariWrapper
    pset.renameArguments(ARG0='chicken_y')
    for i in range(1, 11):
        pset.renameArguments(**{f'ARG{i}': f'car_{i}'})

    return pset