from statsbombpy import sb
import pickle

GOOD_PLAYS = {"carry", "pass", "dribble" , "shot"}
class Offense():
    def __init__(self, match_id , id , index ,type):
        self.match_id = match_id
        self.id = id
        self.index = index
        self.type = type
        self.list_coords = []
        self.list_player_position = []
        self.list_time =[]

    def add_event(self, coords , position , time):
        self.list_coords.append(coords)
        self.list_player_position.append(position)
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
                position = data_dict[i]["type"]["name"]
                min, sec = data_dict[i]["minute"], data_dict[i]["second"]
                self.add_event(coords=coord, position=position, time=min*60+sec)
        self.normalize_timestamp()

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
            play_make=data.get("play_pattern").get("name")
            start_index = PLAY_MAP.get(play_make, PLAY_MAP.get("default"))(data.get("index"), json_data)
            if start_index == -1:
                continue
            offense = Offense(
                match_id = json_data[0].get("match_id"),
                id = data.get("id"),
                index = data.get("index"),
                type = data.get("type").get("name")
            )
            offense.create_offense(start_index, end = data.get("index"), data_dict = json_data)
            list_offense.append(offense)


def main():
    list_offense = []

    competitions = sb.competitions()
    for indexC, rowC in competitions.iterrows():
        matches=sb.matches(competition_id = rowC["competition_id"], season_id = rowC["season_id"])
        for indexM, rowM in matches.iterrows():
            match_id=rowM['match_id']
            events=sb.events(match_id = match_id, fmt = "json")
            list_of_dict=[]
            for d in events.keys():
                list_of_dict.append(events[d])
            process(list_of_dict ,list_offense)

    with open('/data/final_data.pkl', 'wb') as file:
        pickle.dump(list_offense, file)

if __name__ == "__main__":
    main()
