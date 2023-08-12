
from data_process import Get_Offense
import pandas as pd
from mplsoccer import Pitch
import matplotlib.pyplot as plt

class Offense():
    def __init__(self,get_offense : Get_Offense):
        self.match_id = get_offense.match_id
        self.id = get_offense.id
        self.index = get_offense.index
        self.play_pattern = get_offense.play_pattern
        self.end_location = get_offense.end_location
        self.outcome = get_offense.outcome
        self.list_coords = get_offense.list_coords
        self.list_action_type = get_offense.list_action_type
        self.list_time = get_offense.list_time
        self.index_in_current_list = -1


    def print_offense(self):
        df = pd.DataFrame({'match_id' : self.match_id,
                           'id': self.id,
                           'play_pattern': self.play_pattern,
                           'coords': self.list_coords,
                           'player_position': self.list_action_type,
                           'time': self.list_time
                          })
        return df

    def plot_offense(self):
        attack_coords = self.list_coords
        lines = []
        for i in range(len(attack_coords)):
            if i + 1 != len(attack_coords):
                line = [attack_coords[i], attack_coords[i + 1]]
                lines.append(line)
        p = Pitch(pitch_type='statsbomb')
        fig, ax = p.draw(figsize=(8, 8))
        p.scatter(x=[coords[0] for coords in attack_coords], y=[coords[1] for coords in attack_coords], ax=ax)
        p.lines(xstart=[coords[0][0] for coords in lines],
                ystart=[coords[0][1] for coords in lines],
                xend=[coords[1][0] for coords in lines],
                yend=[coords[1][1] for coords in lines],
                ax=ax)
        plt.title(f"Pattern:{self.play_pattern}  Shot Time:{self.list_time[-1]}")
        plt.show()