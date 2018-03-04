import argparse
import time
from enum import Enum

import numpy as np
import visdom

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection, WebSocketConnection  # noqa: F401
from udacidrone.messaging import MsgID


class States(Enum):
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    WAYPOINT = 3
    LANDING = 4
    DISARMING = 5
    STABILIZE = 6


class BackyardFlyer(Drone):

    def __init__(self, connection, plot):
        super().__init__(connection)
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.all_waypoints = self.calculate_box()
        self.in_mission = True
        self.check_state = {}

        # prepare plotter
        if plot:
            self.v = visdom.Visdom()
            assert self.v.check_connection()

            ne = np.array(self.local_position[:2]).reshape(-1, 2)
            self.ne_plot = self.v.scatter(ne, opts=dict(
                title="Local position (north, east)",
                xlabel='North',
                ylabal='East'
            ))

            self.register_callback(MsgID.LOCAL_POSITION, self.update_ne_plot)

        # initial state
        self.flight_state = States.MANUAL

        # TODO: Register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def update_ne_plot(self):
        ne = np.array(self.local_position[:2]).reshape(-1, 2)
        self.v.scatter(ne, win=self.ne_plot, update='append')

    def local_position_callback(self):
        """
        This triggers when `MsgID.LOCAL_POSITION` is received and self.local_position contains new data
        """
        if self.flight_state == States.WAYPOINT:
            # Set the next waypoint when we are just about to reach our current waypoint
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 0.35:
                # We've reached destination, stabilize before going to next waypoint
                self.stabilize_transition()

    def velocity_callback(self):
        """
        This triggers when `MsgID.LOCAL_VELOCITY` is received and self.local_velocity contains new data
        """
        if self.flight_state == States.STABILIZE:
            # TODO More experimentation to figure out if drone is stable
            if np.all(self.local_velocity < .025):
                if len(self.all_waypoints) == 0:
                    self.landing_transition()
                else:
                    self.waypoint_transition()
        elif self.flight_state == States.LANDING:
            if abs(self.global_position[2]) < 0.1:
                self.disarming_transition()

    def state_callback(self):
        """
        This triggers when `MsgID.STATE` is received and self.armed and self.guided contain new data
        """
        if not self.in_mission:
            return
        if self.flight_state == States.MANUAL:
            self.arming_transition()
        elif self.flight_state == States.ARMING:
            if self.armed:
                self.takeoff_transition()
        elif self.flight_state == States.TAKEOFF:
            if self.armed:
                self.waypoint_transition()
        elif self.flight_state == States.DISARMING:
            if not self.armed:
                self.manual_transition()

    def calculate_box(self):
        """
        1. Return waypoints to fly a box. An array of arrays containing deltas.
        """
        home_x = self.global_home[0]
        home_y = self.global_home[1]

        return [
            [home_x, home_y],
            [0.0, home_y + 10.0],
            [home_x + 10.0, home_y + 10.0],
            [home_x + 10.0, 0.0]
        ]

    def arming_transition(self):
        print("arming transition")

        # fix from https://github.com/udacity/fcnd-issue-reports/issues/96
        if self.global_position[0] == 0.0 and self.global_position[1] == 0.0:
            print("no global position data, wait")
            return

        self.take_control()
        self.arm()
        self.set_home_position(self.global_position[0],
                               self.global_position[1],
                               self.global_position[2])

        self.flight_state = States.ARMING

    def takeoff_transition(self):
        print("takeoff transition")

        target_altitude = 3.0
        self.target_position[2] = target_altitude
        self.takeoff(target_altitude)

        self.flight_state = States.TAKEOFF

    def stabilize_transition(self):
        print("stabilize transition")

        self.flight_state = States.STABILIZE

    def waypoint_transition(self):
        """
        1. Command the next waypoint position
        2. Transition to WAYPOINT state
        """
        new_target = self.all_waypoints.pop()

        print("waypoint transition", new_target)

        self.target_position[0] = new_target[0]
        self.target_position[1] = new_target[1]

        self.cmd_position(self.target_position[0],
                          self.target_position[1],
                          self.target_position[2],
                          0)

        self.flight_state = States.WAYPOINT

    def landing_transition(self):
        print("landing transition")
        self.land()
        self.flight_state = States.LANDING

    def disarming_transition(self):
        print("disarm transition")
        self.disarm()
        self.flight_state = States.DISARMING

    def manual_transition(self):
        """This method is provided

        1. Release control of the drone
        2. Stop the connection (and telemetry log)
        3. End the mission
        4. Transition to the MANUAL state
        """
        print("manual transition")

        self.release_control()
        self.stop()
        self.in_mission = False
        self.flight_state = States.MANUAL

    def start(self):
        """This method is provided

        1. Open a log file
        2. Start the drone connection
        3. Close the log file
        """
        print("Creating log file")
        self.start_log("Logs", "NavLog.txt")
        print("starting connection")
        self.connection.start()
        print("Closing log file")
        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    parser.add_argument('--plot', type=bool,help="Enable plotting to local visdom server")
    args = parser.parse_args()

    test = [0.02, 0.02, 0.02, 0.015, 0.015, 0.015]
    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), threaded=False, PX4=False)
    # conn = WebSocketConnection('ws://{0}:{1}'.format(args.host, args.port))
    drone = BackyardFlyer(conn, args.plot)
    time.sleep(2)
    drone.start()
