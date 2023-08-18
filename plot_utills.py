from class_offfense import Offense, Get_Offense
import matplotlib.pyplot as plt
from mplsoccer import Pitch
def data_for_plot(offense):
    """
    this function generates the data structres for plotting the attacks
    :param offense: object from type Offense
    :return: x_points, y_points, lines, x_shot, y_shot
    """
    offense_coords = offense.list_coords
    lines = []
    for i in range(len(offense_coords)):
        if i + 1 != len(offense_coords):
            line = [offense_coords[i], offense_coords[i + 1]]
            lines.append(line)
    x_points = [coords[0] for coords in offense_coords[:-1]]
    y_points = [coords[1] for coords in offense_coords[:-1]]
    x_shot = offense_coords[-1][0]
    y_shot = offense_coords[-1][1]
    return x_points, y_points, lines, x_shot, y_shot


def offense_grid(list_offense, nrows=4, ncols=4, title_text=None):
    pitch = Pitch()
    fig, axs = pitch.grid(nrows=nrows, ncols=ncols, figheight=40,
                          endnote_height=0., space=0.06,
                          axis=False,
                          title_height=0.07, grid_height=0.84)
    axs['title'].text(0.5, 0.65, title_text, fontsize=40, va='center', ha='center')

    for idx, ax in enumerate(axs['pitch'].flat):
        if idx <= len(list_offense) - 1:
            offense = list_offense[idx]
            x_points, y_points, lines, x_shot, y_shot = data_for_plot(offense)
            pitch.scatter(x=x_points, y=y_points, ax=ax)
            pitch.lines(xstart=[coords[0][0] for coords in lines],
                        ystart=[coords[0][1] for coords in lines],
                        xend=[coords[1][0] for coords in lines],
                        yend=[coords[1][1] for coords in lines], ax=ax)
            pitch.scatter(x=x_shot, y=y_shot, marker='football', ax=ax)
