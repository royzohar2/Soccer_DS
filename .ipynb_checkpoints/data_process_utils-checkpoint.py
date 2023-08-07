

class Offense():
    def __init__(self, json_name , id , index ,type):
        self.json_name = json_name
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

    def fix_time_list(self):
        ########################################################################
        pass 

def create_offense(start , end , data_dict, json_name ,type):
    offense = Offense(json_name , data_dict[end]["id"], end , type)
    for i in range(start , end):
        coord =  data_dict[i]["location"]
        position = data_dict[i]["position"]["name"]
        time = data_dict[i]["timestamp"]
        offense.add_event(coords=coord, position=position, time=time)

    offense.fix_time_list()
    return offense