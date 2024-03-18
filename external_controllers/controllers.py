from abc import ABC, abstractmethod 


class ExternalController(ABC):

    @abstractmethod
    def get_accels(self, obs):
        pass

    @abstractmethod
    def reset_controller_state(self):
        pass


class Flatbed(ExternalController):

    def __init__(self, veh_ids):
        self.ka = 2.4
        self.kv = 0.6
        self.kp = 12
        self.d = 5
        self.h = 4

        self.veh_ids = veh_ids

    def get_accels(self, obs):

        accels = {}
        for id in self.veh_ids:
            accel_self = obs[id][0]
            speed_self = obs[id][1]
            speed_front = obs[id][2]
            headway = obs[id][3]
            speed_leader = obs[id][4]

            accels[id] = -self.ka * accel_self + self.kv * (speed_front - speed_self) + self.kp * (headway - self.d - self.h * (speed_self - speed_leader))

        return accels
    
    

class Ploeg(ExternalController):

    def __init__(self, veh_ids):
        self.h = 1
        self.kp = 0.2
        self.kd = 0.7
        self.TS = 0.1

        self.veh_ids = veh_ids

        self.accels = {}
        for id in veh_ids:
            self.accels[id] = 0


    def get_accels(self, obs):

        for id in self.veh_ids:
            speed_self = obs[id][0]
            speed_front = obs[id][1]
            accel_self = obs[id][2]
            accel_front = obs[id][3]
            headway = obs[id][4]


            self.accels[id] += (1 / self.h * (-self.accels[id] + 
                            self.kp * (headway - (2 + self.h * speed_self))+
                            self.kd * (speed_front - speed_self - self.h * accel_self)+
                            accel_front)) * self.TS 
        
        return self.accels
    
    def reset_controller_state(self):
        for id in self.veh_ids:
            self.accels[id] = 0

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