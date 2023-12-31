from statsbombpy import sb
import pandas as pd
import pickle
from mplsoccer import Pitch
import matplotlib.pyplot as plt

GOOD_PLAYS = {"carry", "pass", "dribble" , "shot"}
class Get_Offense():
    def __init__(self, match_id , id , index , play_pattern , end_location , outcome):
        self.match_id = match_id
        self.id = id
        self.index = index
        self.play_pattern = play_pattern
        self.end_location = end_location
        self.outcome = outcome
        self.list_coords = []
        self.list_action_type = []
        self.list_time =[]

    def add_event(self, coords , action_type , time):
        self.list_coords.append(coords)
        self.list_action_type.append(action_type)
        self.list_time.append(time)

    def normalize_timestamp(self):
        try:
            first = self.list_time[0]
            for i in range(len(self.list_time)):
                self.list_time[i] -= first
        except Exception as e:
            pass

    def create_offense(self, start , end , data_dict):
        for i in range(start, end + 1):
            current_play = data_dict[i].get("type").get("name").lower()
            if current_play in GOOD_PLAYS:
                coord = data_dict[i]["location"]
                action_type = data_dict[i]["type"]["name"]
                min, sec = data_dict[i]["minute"], data_dict[i]["second"]
                self.add_event(coords=coord, action_type=action_type, time=min*60+sec)
        self.normalize_timestamp()

# maybe we should delete this
    def print_offense(self):
        df = pd.DataFrame({'match_id' : self.match_id,
                           'id': self.id,
                           'play_pattern': self.play_pattern,
                           'coords': self.list_coords[:-1],
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

#########################################################################################################


def from_corner_handler(shot_index, data: any):
    i = shot_index - 1
    while True:
        try:
            if data[i].get("pass").get("type").get("name") != "Corner":
                i -= 1
            else:
                return i
        except:
            i -= 1


def free_kick_handler(shot_index, data:any):
    i = shot_index - 1
    while data[i].get("type", None).get("name") != "Foul Won":
        i -= 1
    return i


def throw_in_handler(shot_index, data: any):
    i = shot_index - 1
    while True:
        try:
            if data[i].get("pass").get("type").get("name") != "Throw-in":
                i-=1
            else:
                return i
        except:
            i-=1


def goal_keeper_handler(shot_index, data: any):
    i = shot_index - 1
    while True:
        try:
            if data[i].get("position").get("name") != "Goalkeeper":
                i-=1
            else:
                return i
        except:
            i-=1


def from_keeper_handler(shot_index, data: any):
    i = shot_index - 1
    while True:
        try:
            if data[i].get("position").get("name") != "Goalkeeper":
                i-=1
            else:
                return i
        except:
            i-=1

def from_counter_handler(shot_index, data: any):
    i=shot_index - 1
    possession_team = data[i].get("possession_team").get("name")
    while True:
        try:
            if data[i].get("possession_team").get("name") == possession_team:
                i-=1
            else:
                return i + 1
        except:
            i-=1

def regular_play_handler(shot_index, data: any):
    i=shot_index - 1
    possession_team = data[i].get("possession_team").get("name")
    while True:
        try:
            if (data[i].get('type').get('name') in {'Carry', 'Pass', 'Dribble','Ball Receipt*'}) and (data[i].get('team').get('name') != possession_team):
                return i + 1
            else:
                i-=1
        except:
            i-=1


def default_handler(shot_index, data: any):
    return -1


PLAY_MAP={
    "From Corner": from_corner_handler,
    "From Free Kick": free_kick_handler,
    "From Throw In": throw_in_handler,
    "From Goal Kick": goal_keeper_handler,
    "From Counter": from_counter_handler,
    "From Keeper": from_keeper_handler,
    "Regular Play": regular_play_handler,
    "default": default_handler
}


def process(json_data , list_offense):

    for data in json_data:
        curr_type = data.get("type").get("name")
        if curr_type == "Shot":
            shot_end_location = data.get("shot").get("end_location")
            outcome = data.get("shot").get("outcome").get("name")
            play_make=data.get("play_pattern").get("name")
            start_index = PLAY_MAP.get(play_make, PLAY_MAP.get("default"))(data.get("index"), json_data)
            if start_index == -1:
                continue
            offense = Get_Offense(
                match_id = json_data[0].get("match_id"),
                id = data.get("id"),
                index = data.get("index"),
                play_pattern=play_make ,
                end_location = shot_end_location , 
                outcome = outcome,
            )
            offense.create_offense(start_index, end = data.get("index"), data_dict = json_data)
            list_offense.append(offense)


def main():
    list_offense = []

    competitions = sb.competitions()
    for indexC, rowC in competitions.iterrows():
        try:
            matches=sb.matches(competition_id = rowC["competition_id"], season_id = rowC["season_id"])
            for indexM, rowM in matches.iterrows():
                try:
                    match_id=rowM['match_id']
                    events=sb.events(match_id = match_id, fmt = "json")
                    list_of_dict=[]
                    for d in events.keys():
                        list_of_dict.append(events[d])
                    process(list_of_dict ,list_offense)
                except:
                    print("small try")
        except:
            print("big try")

    with open('data/final_data.pkl', 'wb') as file:
        pickle.dump(list_offense, file)

if __name__ == "__main__":
    main()
