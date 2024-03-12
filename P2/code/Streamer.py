# Copyright © 2018 Naturalpoint
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish a connection (by creating a NatNetClient),
# and receive data via a NatNet connection and decode it using the NatNetClient library.

import sys
import time
import DataDescriptions
import MoCapData
import numpy as np

import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import axes3d

from NatNetClient import NatNetClient
from pythonosc import udp_client
from collections import deque

# constant parameters for calculations
HAND_MAX_DISTANCE = 1.5  # 1.8
HEAD_ROT_MAX_ANGLE = 45
HEAD_FOLD_MAX_ANGLE = 55
HEAD_FOLD_MIN_ANGLE = 15
SPINE_FOLD_MAX_ANGLE = 30  # 35
SPINE_FOLD_MIN_ANGLE = 8
PEDAL_THRESHOLD = 0.1  # revise?
HEAD_PROXIMITY_THRESHOLD = 0.07
MARKERS_THRESHOLD = 0.5

LH_ID = 11
RH_ID = 12
HEAD_ID = 13
BACK_ID = 14
# FHEAD_ID = 13
# BHEAD_ID = 14
# LFOOT_ID = 15
# RFOOT_ID = 16

MARKERSET = np.zeros((3, 3))
LH_POS = [0, 0, 0]
RH_POS = [0, 0, 0]
FHEAD_POS = [0, 0, 0]
BHEAD_POS = [0, 0, 0]
LFOOT_VAL = 0
RFOOT_VAL = 0
_is_first_receive = True

# Initialize a 3D numpy array buffer: markers x axes x samples


class Streamer:
    def __init__(self):
        self.send_client = None
        self.sd = None
        self.prev_meanpos = None
        self.jump_compansate_threshold_count = 0

    # calculation functions
    def head_rotation(self, fhead, bhead):
        """calculates the head's angle from the Z axis on XZ plane,
        based on two markers placed on forehead and the back of the head

        returns
        a <= -threshold : 0
        -threshold < a < threshold : (0, 1)
        a >=  threshold : 1
        """
        v = np.subtract(fhead, bhead)
        if np.sqrt(v.dot(v)) < HEAD_PROXIMITY_THRESHOLD:
            angle = 0
        else:
            v[1] = 0  # projecting the vector to XZ plane
            v[2] = np.abs(v[2])  # mirroring on XY plane
            v_ref = [0, 0, 1]
            angle = np.degrees(
                np.arccos(
                    np.dot(v, v_ref) / (np.linalg.norm(v) * np.linalg.norm(v_ref))
                )
            )
            if np.cross(v, v_ref)[1] < 0:
                angle = -angle
        return np.clip(angle / HEAD_ROT_MAX_ANGLE, -1, 1)

    def spine_fold(self, spine1, spine2, spine3, head):
        """calculates curvature amount of the upper body and head

        returns
        a <= -threshold : 0
        -threshold < a < threshold : (0, 1)
        a >=  threshold : 1
        """
        # print(['an', spine1, spine2, spine3])
        lower_body_vector = spine2 - spine1
        upper_body_vector = spine3 - spine2
        head_vector = head - spine3
        angle_spine = np.degrees(
            np.arccos(
                np.dot(lower_body_vector, upper_body_vector)
                / (
                    np.linalg.norm(lower_body_vector)
                    * np.linalg.norm(upper_body_vector)
                )
            )
        )
        # if np.cross(lower_body_vector, upper_body_vector)[1] > 0:
        #     angle_spine *= -1
        angle_head = np.degrees(
            np.arccos(
                np.dot(upper_body_vector, head_vector)
                / (np.linalg.norm(upper_body_vector) * np.linalg.norm(head_vector))
            )
        )
        # if np.cross(upper_body_vector, head_vector)[1] > 0:
        #     angle_head *= -1
        # input([angle_spine, angle_head])
        return (
            angle_spine,
            angle_head,
            (angle_spine - SPINE_FOLD_MIN_ANGLE)
            / (SPINE_FOLD_MAX_ANGLE - SPINE_FOLD_MIN_ANGLE),
            (angle_head - HEAD_FOLD_MIN_ANGLE)
            / (HEAD_FOLD_MAX_ANGLE - HEAD_FOLD_MIN_ANGLE),
        )

    def hand_distance(self, lhand, rhand):
        """Calculate the distance between hands
        Returns distance, and 1 if right hand is higher, 0 if left hand is higher"""
        dist = np.linalg.norm(lhand - rhand)
        dist = np.sqrt(np.sum((lhand - rhand) ** 2, axis=0))
        dist /= HAND_MAX_DISTANCE
        return dist, int(rhand[1] > lhand[1])

    def pedal_control(self, foot):
        """Calculates boolean value of either the toetip of the foot is high or low
        Returns 1 if the the toetip is high, otherwise 0"""
        return int(foot[1] > PEDAL_THRESHOLD)

    def dist(self, p1, p2):
        if p1 is None or p2 is None:
            return 0
        return np.sqrt(np.sum((p2 - p1) ** 2, axis=0))

    # This is a callback function that gets connected to the NatNet client
    # and called once per mocap frame.
    def receive_new_frame(self, data_dict):
        order_list = [
            "frameNumber",
            "markerSetCount",
            "unlabeledMarkersCount",
            "rigidBodyCount",
            "skeletonCount",
            "labeledMarkerCount",
            "timecode",
            "timecodeSub",
            "timestamp",
            "isRecording",
            "trackedModelsChanged",
        ]
        dump_args = False
        if dump_args == True:
            out_string = "    "
            for key in data_dict:
                out_string += key + "="
                if key in data_dict:
                    out_string += data_dict[key] + " "
                out_string += "/"
            print(out_string)

    # This is a callback function that gets connected to the NatNet client
    # and called once per labeled marker per <?>.
    def receive_labeled_marker(self, id, position, size):
        # print("Received labeled marker", id, position, size)
        # Create an OSC message and send it
        message = f"/{id}pos"
        args = position
        self.send_client.send_message(message, args)
        # message = f"/{id}size"
        # args = size
        # send_client.send_message(message, args)

    def receive_skeleton(self, id, skeleton):
        for rb in skeleton.rigid_body_list:
            pass  # receive_rigid_body_frame(id, rb.pos, rb.rot)

    def append_buffer(self, new_data):
        self.sd.data_buffer = np.roll(self.sd.data_buffer, -1, axis=0)
        self.sd.data_buffer[-1] = new_data

    # This is a callback function that gets connected to the NatNet client
    # and called once per labeled marker per <?>.
    def receive_labeled_marker(self, labeled_marker_data):

        received_markers = np.array(labeled_marker_data.labeled_marker_list)

        # list of positions
        list_pos = np.array([m.pos for m in received_markers])
        if list_pos.ndim == 1:
            return

        # y-z correction, swap
        list_pos[:, [1, 2]] = list_pos[:, [2, 1]]

        meanpos = np.mean(list_pos, axis=0)
        if self.dist(meanpos, self.prev_meanpos) > 0.25:
            # print(self.dist(meanpos, self.prev_meanpos))
            self.jump_compansate_threshold_count += 1
            if self.jump_compansate_threshold_count < 120:
                meanpos = np.copy(self.prev_meanpos)
            else:
                self.jump_compansate_threshold_count = 0

        self.sd.x = meanpos[0]
        self.sd.y = meanpos[1]
        self.sd.z = meanpos[2]

        self.sd.list_pos = list_pos
        self.sd.meanpos = meanpos
        self.append_buffer(meanpos)

        self.prev_meanpos = np.copy(meanpos)

        # message = "/poses"
        # args = [spineang, headang, spinefold, headfold]
        # self.send_client.send_message(message, [self.sd.x, self.sd.y])

        # spine_markers = list_pos[list_pos[:, 1] > MARKERS_THRESHOLD]

        # global _is_first_receive
        # global MARKERSET
        # if len(spine_markers) > 2:  # if all markers received
        #     if _is_first_receive:  # if first time/reset, init markers
        #         # print('here')
        #         spine_markers = spine_markers[np.argsort(spine_markers[:, 1])]
        #         MARKERSET = spine_markers
        #         # print(MARKERSET)
        #         _is_first_receive = False
        #     else:  # if not first time/reset assign markerset based on proximity
        #         # print('futzk')
        #         for i, dot in enumerate(MARKERSET):
        #             closest = np.argmin(self.dist(dot, spine_markers))
        #             MARKERSET[i] = spine_markers[closest]
        # elif not spine_markers.any():  # if not all markers received, set reset flag
        #     _is_first_receive = True

        # if (MARKERSET[0] == MARKERSET[1]).all() or (
        #     MARKERSET[1] == MARKERSET[2]
        # ).all():  # if marker pairs assigned the same point
        #     _is_first_receive = True

        # # headrot = head_rotation(spine_markers[0], spine_markers[1])
        # spineang, headang, spinefold, headfold = self.spine_fold(
        #     MARKERSET[0], MARKERSET[1], MARKERSET[2], BHEAD_POS
        # )
        # if not self.spine_fold:  # if spinefold returned nan
        #     _is_first_receive = True

        # # print('markerset', MARKERSET)
        # # print('head2', BHEAD_POS)
        # # handdist, handact = hand_distance(spine_markers[0], spine_markers[1])
        # # pedal = pedal_control(spine_markers[-1])

        # message = "/fold"
        # args = [spineang, headang, spinefold, headfold]
        # self.send_client.send_message(message, args)
        # # print(message, args)

    # This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
    def receive_rigid_body_frame(self, new_id, position, rotation):
        # print("Received frame for rigid body", new_id)
        # print( "Received frame for rigid body", new_id," ",position," ",rotation )
        global FHEAD_POS
        global BHEAD_POS
        global LH_POS
        global RH_POS
        global LFOOT_VAL
        global RFOOT_VAL

        def send_handdist():
            handdist, handact = self.hand_distance(np.array(LH_POS), np.array(RH_POS))
            message = f"/handdist"
            args = [handdist, handact]
            self.send_client.send_message(message, args)

        def send_headrot():
            headrot = self.head_rotation(FHEAD_POS, BHEAD_POS)
            message = f"/headrot"
            args = headrot
            self.send_client.send_message(message, args)

        position = [position[0], position[2], position[1]]  # y-z correction, swap

        if new_id == LH_ID:
            self.sd.lh_pos = position
        if new_id == RH_ID:
            self.sd.rh_pos = position
        if new_id == HEAD_ID:
            self.sd.headpos = position
        if new_id == BACK_ID:
            self.sd.backpos = position
        # elif new_id == BHEAD_ID:
        #     self.sd.bhead_pos = position
        # elif new_id == LFOOT_ID:
        #     lpedal = self.pedal_control(position)
        #     if lpedal != LFOOT_VAL:
        #         message = f"/lpedal"
        #         self.send_client.send_message(message, lpedal)
        #         LFOOT_VAL = lpedal
        # elif new_id == RFOOT_ID:
        #     rpedal = self.pedal_control(position)
        #     if rpedal != RFOOT_VAL:
        #         message = f"/rpedal"
        #         self.send_client.send_message(message, rpedal)
        #         RFOOT_VAL = rpedal
        # message = f"/{new_id}pos"
        # args = position
        # self.send_client.send_message(message, args)

        meanpos = np.mean(
            [self.sd.lhpos, self.sd.rhpos, self.sd.headpos, self.sd.backpos], axis=0
        )
        self.sd.x = meanpos[0]
        self.sd.y = meanpos[1]
        self.sd.z = meanpos[2]

        self.append_buffer(meanpos)

    def add_lists(self, totals, totals_tmp):
        totals[0] += totals_tmp[0]
        totals[1] += totals_tmp[1]
        totals[2] += totals_tmp[2]
        return totals

    def print_configuration(self, natnet_client):
        print("Connection Configuration:")
        print("  Client:          %s" % natnet_client.local_ip_address)
        print("  Server:          %s" % natnet_client.server_ip_address)
        print("  Command Port:    %d" % natnet_client.command_port)
        print("  Data Port:       %d" % natnet_client.data_port)

        if natnet_client.use_multicast:
            print("  Using Multicast")
            print("  Multicast Group: %s" % natnet_client.multicast_address)
        else:
            print("  Using Unicast")

        # NatNet Server Info
        application_name = natnet_client.get_application_name()
        nat_net_requested_version = natnet_client.get_nat_net_requested_version()
        nat_net_version_server = natnet_client.get_nat_net_version_server()
        server_version = natnet_client.get_server_version()

        print("  NatNet Server Info")
        print("    Application Name %s" % (application_name))
        print(
            "    NatNetVersion  %d %d %d %d"
            % (
                nat_net_version_server[0],
                nat_net_version_server[1],
                nat_net_version_server[2],
                nat_net_version_server[3],
            )
        )
        print(
            "    ServerVersion  %d %d %d %d"
            % (
                server_version[0],
                server_version[1],
                server_version[2],
                server_version[3],
            )
        )
        print("  NatNet Bitstream Requested")
        print(
            "    NatNetVersion  %d %d %d %d"
            % (
                nat_net_requested_version[0],
                nat_net_requested_version[1],
                nat_net_requested_version[2],
                nat_net_requested_version[3],
            )
        )
        # print("command_socket = %s"%(str(natnet_client.command_socket)))
        # print("data_socket    = %s"%(str(natnet_client.data_socket)))

    def print_commands(self, can_change_bitstream):
        outstring = "Commands:\n"
        outstring += "Return Data from Motive\n"
        outstring += "  s  send data descriptions\n"
        outstring += "  r  resume/start frame playback\n"
        outstring += "  p  pause frame playback\n"
        outstring += "     pause may require several seconds\n"
        outstring += "     depending on the frame data size\n"
        outstring += "Change Working Range\n"
        outstring += (
            "  o  reset Working Range to: start/current/end frame = 0/0/end of take\n"
        )
        outstring += "  w  set Working Range to: start/current/end frame = 1/100/1500\n"
        outstring += "Return Data Display Modes\n"
        outstring += (
            "  j  print_level = 0 supress data description and mocap frame data\n"
        )
        outstring += "  k  print_level = 1 show data description and mocap frame data\n"
        outstring += "  l  print_level = 20 show data description and every 20th mocap frame data\n"
        outstring += "Change NatNet data stream version (Unicast only)\n"
        outstring += "  3  Request 3.1 data stream (Unicast only)\n"
        outstring += "  4  Request 4.0 data stream (Unicast only)\n"
        outstring += "t  data structures self test (no motive/server interaction)\n"
        outstring += "c  show configuration\n"
        outstring += "h  print commands\n"
        outstring += "q  quit\n"
        outstring += "\n"
        outstring += "NOTE: Motive frame playback will respond differently in\n"
        outstring += "       Endpoint, Loop, and Bounce playback modes.\n"
        outstring += "\n"
        outstring += (
            "EXAMPLE: PacketClient [serverIP [ clientIP [ Multicast/Unicast]]]\n"
        )
        outstring += '         PacketClient "192.168.10.14" "192.168.10.14" Multicast\n'
        outstring += '         PacketClient "127.0.0.1" "127.0.0.1" u\n'
        outstring += "\n"
        print(outstring)

    def request_data_descriptions(self, s_client):
        # Request the model definitions
        s_client.send_request(
            s_client.command_socket,
            s_client.NAT_REQUEST_MODELDEF,
            "",
            (s_client.server_ip_address, s_client.command_port),
        )

    def test_classes(self):
        totals = [0, 0, 0]
        print("Test Data Description Classes")
        totals_tmp = DataDescriptions.test_all()
        totals = self.add_lists(totals, totals_tmp)
        print("")
        print("Test MoCap Frame Classes")
        totals_tmp = MoCapData.test_all()
        totals = self.add_lists(totals, totals_tmp)
        print("")
        print("All Tests totals")
        print("--------------------")
        print("[PASS] Count = %3.1d" % totals[0])
        print("[FAIL] Count = %3.1d" % totals[1])
        print("[SKIP] Count = %3.1d" % totals[2])

    def my_parse_args(self, arg_list, args_dict):
        # set up base values
        arg_list_len = len(arg_list)
        if arg_list_len > 1:
            args_dict["serverAddress"] = arg_list[1]
            if arg_list_len > 2:
                args_dict["clientAddress"] = arg_list[2]
            if arg_list_len > 3:
                if len(arg_list[3]):
                    args_dict["use_multicast"] = True
                    if arg_list[3][0].upper() == "U":
                        args_dict["use_multicast"] = False

        return args_dict

    def start_data_acquisition(self, sd):

        self.sd = sd

        # Set up the OSC client
        # send_ip_address = "127.0.0.1"
        # send_port = 8000
        # self.send_client = udp_client.SimpleUDPClient(send_ip_address, send_port)

        optionsDict = {}
        optionsDict["clientAddress"] = "169.254.63.12"
        optionsDict["serverAddress"] = "169.254.228.107"  # "129.240.79.163"
        # optionsDict["clientAddress"] = "127.0.0.1"
        # optionsDict["serverAddress"] = "127.0.0.1"
        optionsDict["use_multicast"] = True

        # This will create a new NatNet client
        optionsDict = self.my_parse_args(sys.argv, optionsDict)

        streaming_client = NatNetClient()
        streaming_client.set_client_address(optionsDict["clientAddress"])
        streaming_client.set_server_address(optionsDict["serverAddress"])
        streaming_client.set_use_multicast(optionsDict["use_multicast"])

        # Configure the streaming client to call our rigid body handler on the emulator to send data out.
        streaming_client.new_frame_listener = self.receive_new_frame
        streaming_client.rigid_body_listener = self.receive_rigid_body_frame
        streaming_client.labeled_marker_listener = self.receive_labeled_marker
        streaming_client.skeleton_listener = self.receive_skeleton

        # Start up the streaming client now that the callbacks are set up.
        # This will run perpetually, and operate on a separate thread.
        is_running = streaming_client.run()
        if not is_running:
            print("ERROR: Could not start streaming client.")
            try:
                sys.exit(1)
            except SystemExit:
                print("...")
            finally:
                print("exiting")

        is_looping = True
        time.sleep(1)
        if streaming_client.connected() is False:
            print(
                "ERROR: Could not connect properly.  Check that Motive streaming is on."
            )
            try:
                sys.exit(2)
            except SystemExit:
                print("...")
            finally:
                print("exiting")

        self.print_configuration(streaming_client)
        print("\n")
        self.print_commands(streaming_client.can_change_bitstream_version())

        while is_looping:
            inchars = input("Enter command or ('h' for list of commands)\n")
            if len(inchars) > 0:
                c1 = inchars[0].lower()
                if c1 == "h":
                    self.print_commands(streaming_client.can_change_bitstream_version())
                elif c1 == "c":
                    self.print_configuration(streaming_client)
                elif c1 == "s":
                    self.request_data_descriptions(streaming_client)
                    time.sleep(1)
                elif (c1 == "3") or (c1 == "4"):
                    if streaming_client.can_change_bitstream_version():
                        tmp_major = 4
                        tmp_minor = 0
                        if c1 == "3":
                            tmp_major = 3
                            tmp_minor = 1
                        return_code = streaming_client.set_nat_net_version(
                            tmp_major, tmp_minor
                        )
                        time.sleep(1)
                        if return_code == -1:
                            print(
                                "Could not change bitstream version to %d.%d"
                                % (tmp_major, tmp_minor)
                            )
                        else:
                            print("Bitstream version at %d.%d" % (tmp_major, tmp_minor))
                    else:
                        print("Can only change bitstream in Unicast Mode")

                elif c1 == "p":
                    sz_command = "TimelineStop"
                    return_code = streaming_client.send_command(sz_command)
                    time.sleep(1)
                    print("Command: %s - return_code: %d" % (sz_command, return_code))
                elif c1 == "r":
                    sz_command = "TimelinePlay"
                    return_code = streaming_client.send_command(sz_command)
                    print("Command: %s - return_code: %d" % (sz_command, return_code))
                elif c1 == "o":
                    tmpCommands = [
                        "TimelinePlay",
                        "TimelineStop",
                        "SetPlaybackStartFrame,0",
                        "SetPlaybackStopFrame,1000000",
                        "SetPlaybackLooping,0",
                        "SetPlaybackCurrentFrame,0",
                        "TimelineStop",
                    ]
                    for sz_command in tmpCommands:
                        return_code = streaming_client.send_command(sz_command)
                        print(
                            "Command: %s - return_code: %d" % (sz_command, return_code)
                        )
                    time.sleep(1)
                elif c1 == "w":
                    tmp_commands = [
                        "TimelinePlay",
                        "TimelineStop",
                        "SetPlaybackStartFrame,10",
                        "SetPlaybackStopFrame,1500",
                        "SetPlaybackLooping,0",
                        "SetPlaybackCurrentFrame,100",
                        "TimelineStop",
                    ]
                    for sz_command in tmp_commands:
                        return_code = streaming_client.send_command(sz_command)
                        print(
                            "Command: %s - return_code: %d" % (sz_command, return_code)
                        )
                    time.sleep(1)
                elif c1 == "t":
                    self.test_classes()

                elif c1 == "j":
                    streaming_client.set_print_level(0)
                    print(
                        "Showing only received frame numbers and supressing data descriptions"
                    )
                elif c1 == "k":
                    streaming_client.set_print_level(1)
                    print("Showing every received frame")

                elif c1 == "l":
                    print_level = streaming_client.set_print_level(20)
                    print_level_mod = print_level % 100
                    if print_level == 0:
                        print(
                            "Showing only received frame numbers and supressing data descriptions"
                        )
                    elif print_level == 1:
                        print("Showing every frame")
                    elif print_level_mod == 1:
                        print("Showing every %dst frame" % print_level)
                    elif print_level_mod == 2:
                        print("Showing every %dnd frame" % print_level)
                    elif print_level == 3:
                        print("Showing every %drd frame" % print_level)
                    else:
                        print("Showing every %dth frame" % print_level)

                elif c1 == "q":
                    is_looping = False
                    streaming_client.shutdown()
                    break
                else:
                    print("Error: Command %s not recognized" % c1)
                print("Ready...\n")
        print("exiting")
