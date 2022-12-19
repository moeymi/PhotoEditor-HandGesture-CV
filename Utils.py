class hand_helper:
    def __init__(self):
        None
    
    def __calculate_hand_size(self):
        None
    
    def calculate_translation_normalized(self, starting_point, current_point, frame_shape):
        return ((current_point[0] - starting_point[0]) / frame_shape[1], 
                (current_point[1] - starting_point[1]) / frame_shape[0])