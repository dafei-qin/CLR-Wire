
def check_contain_surface(surface_list, surface_type: str):
    counter = 0 
    for surface in surface_list:
        if surface['type'] == surface_type:
            counter += 1
    return counter
