import operator
import random
from deap import gp

def protected_div(left, right):
    if abs(right) < 1e-6: return 1
    return left / right

def if_then_else(input, output1, output2):
    return output1 if input else output2

def random_101():
    return random.randint(-1, 1)

def setup_space_invaders_primitives():
    """
    Primitive specifiche per Space Invaders (8 input).
    """
    # 8 Input come definito nel wrapper
    pset = gp.PrimitiveSet("MAIN", 8)
    
    # Operatori Standard
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(max, 2)
    pset.addPrimitive(min, 2)
    pset.addPrimitive(operator.lt, 2)
    pset.addPrimitive(operator.gt, 2)
    pset.addPrimitive(if_then_else, 3)
    
    pset.addEphemeralConstant("rand101", random_101)
    
    # Rinomina Argomenti (Mapping dal tuo Wrapper)
    # vec[0] = px
    # vec[1] = nearest_alien_x_dist
    # vec[2] = nearest_alien_y_dist
    # vec[3] = bullet_x_dist
    # vec[4] = bullet_y_dist
    # vec[5] = alien_density
    # vec[6] = is_covered
    # vec[7] = is_aiming_alien
    
    pset.renameArguments(ARG0='player_x')
    pset.renameArguments(ARG1='alien_dx')
    pset.renameArguments(ARG2='alien_dy')
    pset.renameArguments(ARG3='bullet_dx')
    pset.renameArguments(ARG4='bullet_dy')
    pset.renameArguments(ARG5='alien_density')
    pset.renameArguments(ARG6='is_shielded')
    pset.renameArguments(ARG7='is_aiming')

    return pset