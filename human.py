
class Human():
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def __str__(self):
        return "Human {}".format(self.player)