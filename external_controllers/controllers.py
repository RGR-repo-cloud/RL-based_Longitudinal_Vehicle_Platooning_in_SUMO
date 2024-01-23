class Fastbed(object):

    def __init__(self):
        self.ka = 2.4
        self.kv = 0.6
        self.kp = 12
        self.d = 5
        self.h = 4


    def get_accel(self, obs):

        accel_self = obs[0]
        speed_self = obs[1]
        speed_front = obs[2]
        headway = obs[3]
        speed_leader = obs[4]

        return -self.ka * accel_self + self.kv * (speed_front - speed_self) + self.kp * (headway - self.d - self.h * (speed_self - speed_leader))