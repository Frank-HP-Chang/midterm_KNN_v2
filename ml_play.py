"""
The template of the main script of the machine learning process
"""
import pickle
import numpy as np
#import games.arkanoid.communication as comm
#from games.arkanoid.communication import ( \
#    SceneInfo, GameStatus, PlatformAction
#)
import os.path as path
from mlgame.communication import ml as comm

def ml_loop(side: "1P"):
    """
    The main loop of the machine learning process

    This loop is run in a separate process, and communicates with the game process.

    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.
    ball_served = False
    filename=path.join(path.dirname(__file__),"save/KNN.pickle")
    #filename = "C:\\Users\\HsuanPu_Chang\\Desktop\\ML_Game_File\\ML_Game_MidtermTwo\\MLGame-beta6.0\\clf_SVMClassification_VectorsAndDirection.pickle"
    with open(filename, 'rb') as file:
        clf = pickle.load(file)

    # 2. Inform the game process that ml process is ready before start the loop.
    comm.ml_ready()
    """
    s = [93,93]
    def get_direction(ball_x,ball_y,ball_pre_x,ball_pre_y):
        VectorX = ball_x - ball_pre_x
        VectorY = ball_y - ball_pre_y
        if(VectorX>=0 and VectorY>=0):
            return 0
        elif(VectorX>0 and VectorY<0):
            return 1
        elif(VectorX<0 and VectorY>0):
            return 2
        elif(VectorX<0 and VectorY<0):
            return 3
        """

    # 3. Start an endless loop.
    while True:
        scene_info = comm.recv_from_game()
        # 3.1. Receive the scene information sent from the game process.
        feature = []
        feature.append(scene_info["ball"])
        feature.append(scene_info["ball_speed"])
        feature.append(scene_info["blocker"])
        feature.append(scene_info["platform_1P"])
        feature=np.array(feature)
        feature=feature.reshape((-1,8))
        print(feature)

        #feature.append(get_direction(feature[0],feature[1],s[0],s[1]))
        #s = [feature[0], feature[1]]
        #feature = np.array(feature)
        #feature = feature.reshape((-1,4))
        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        if scene_info["status"] != "GAME_ALIVE":
            # Do some updating or resetting stuff
            ball_served = False

            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue


        # 3.3. Put the code here to handle the scene information

        # 3.4. Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_LEFT"})
            ball_served = True
        else:
                
            y = clf.predict(feature)
            
            if y <= 0.5:
                comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
                print('NONE')
            elif y > 0.5 and y <= 1.5:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
                print('LEFT')
            elif y > 1.5:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
                print('RIGHT')
