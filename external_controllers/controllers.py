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
    

class Ploeg(object):

    def __init__(self):
        self.h = 1
        self.kp = 0.2
        self.kd = 0.7
        self.TS = 1.0

        self.accel = 0


    def get_accel(self, obs):

        speed_self = obs[0]
        speed_front = obs[1]
        accel_self = obs[2]
        accel_front = obs[3]
        headway = obs[4]


        self.accel += ((1 / self.h * (-self.accel + 
                        self.kp * (headway - (2 + self.h * speed_self))+
                        self.kd * (speed_front - speed_self - self.h * accel_self)+
                        accel_front)) * self.TS )
        
        return self.accel

"""
    double
MSCFModel_CC::_ploeg(const MSVehicle* veh, double egoSpeed, double predSpeed, double predAcceleration, double gap2pred) const {
    CC_VehicleVariables* vars = (CC_VehicleVariables*)veh->getCarFollowVariables();
    return (1 / vars->ploegH * (
                -vars->controllerAcceleration +
                vars->ploegKp * (gap2pred - (2 + vars->ploegH * egoSpeed)) +
                vars->ploegKd * (predSpeed - egoSpeed - vars->ploegH * veh->getAcceleration()) +
                predAcceleration
            )) * TS ;
}

ploegH(0.5), ploegKp(0.2), ploegKd(0.7)
"""