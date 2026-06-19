import cv2
import mediapipe as mp
import numpy as np
import time
import random
import pandas as pd
from datetime import datetime


def run_gaze_experiment():

    # -----------------------------
    # MediaPipe Setup
    # -----------------------------
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    LEFT_IRIS = [474,475,476,477]
    RIGHT_IRIS = [469,470,471,472]

    # -----------------------------
    # Camera
    # -----------------------------
    cap = cv2.VideoCapture(0)

    # -----------------------------
    # Calibration
    # -----------------------------
    calibration = {"LEFT":[], "CENTER":[], "RIGHT":[]}
    calibration_stage = ["LEFT","CENTER","RIGHT"]

    stage_index = 0
    stage_time = time.time()

    print("Calibration started")

    def get_ratio(mesh_points):

        left_iris = mesh_points[LEFT_IRIS]
        right_iris = mesh_points[RIGHT_IRIS]

        left_center = np.mean(left_iris,axis=0)
        right_center = np.mean(right_iris,axis=0)

        left_ratio = (left_center[0]-mesh_points[33][0])/(mesh_points[133][0]-mesh_points[33][0])
        right_ratio = (right_center[0]-mesh_points[362][0])/(mesh_points[263][0]-mesh_points[362][0])

        return (left_ratio+right_ratio)/2

    while stage_index < 3:

        ret,frame = cap.read()
        frame = cv2.flip(frame,1)

        h,w = frame.shape[:2]

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        direction = calibration_stage[stage_index]

        cv2.putText(frame,f"LOOK {direction}",(50,80),
                    cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),3)

        if results.multi_face_landmarks:

            mesh_points = np.array(
                [(int(p.x*w),int(p.y*h))
                 for p in results.multi_face_landmarks[0].landmark]
            )

            ratio = get_ratio(mesh_points)
            calibration[direction].append(ratio)

        if time.time()-stage_time > 3:
            stage_index += 1
            stage_time = time.time()

        cv2.imshow("Calibration",frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # -----------------------------
    # Thresholds
    # -----------------------------
    left_thresh = np.mean(calibration["LEFT"])
    center_thresh = np.mean(calibration["CENTER"])
    right_thresh = np.mean(calibration["RIGHT"])

    print("Calibration complete")

    # -----------------------------
    # Experiment Variables
    # -----------------------------
    trial = 0
    total_trials = 10
    correct = 0

    reaction_times = []
    saccade_speeds = []

    prev_eye_x = None
    prev_time = None

    experiment_data = []

    stimulus_positions = ["LEFT","RIGHT"]

    state = "FIXATION"
    stimulus_time = 0
    fixation_time = time.time()

    print("Experiment Started")

    # -----------------------------
    # Experiment Loop
    # -----------------------------
    while trial < total_trials:

        ret,frame = cap.read()
        frame = cv2.flip(frame,1)

        h,w = frame.shape[:2]

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        gaze = "CENTER"

        # Fixation
        if state == "FIXATION":

            cv2.putText(frame,"+",(w//2,h//2),
                        cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),4)

            if time.time()-fixation_time > random.uniform(1.5,2.5):

                stimulus = random.choice(stimulus_positions)
                stimulus_time = time.time()
                state = "STIMULUS"

        # Stimulus
        if state == "STIMULUS":

            if stimulus == "LEFT":
                cv2.circle(frame,(80,h//2),40,(0,0,255),-1)

            if stimulus == "RIGHT":
                cv2.circle(frame,(w-80,h//2),40,(0,0,255),-1)

        if results.multi_face_landmarks:

            mesh_points = np.array(
                [(int(p.x*w),int(p.y*h))
                 for p in results.multi_face_landmarks[0].landmark]
            )

            ratio = get_ratio(mesh_points)

            if ratio < (left_thresh+center_thresh)/2:
                gaze="LEFT"
            elif ratio > (right_thresh+center_thresh)/2:
                gaze="RIGHT"
            else:
                gaze="CENTER"

            # -----------------------------
            # Saccade Speed
            # -----------------------------
            eye_x = np.mean([mesh_points[474][0], mesh_points[469][0]])
            current_time = time.time()

            speed = 0

            if prev_eye_x is not None:

                distance = abs(eye_x-prev_eye_x)
                dt = current_time-prev_time

                if dt > 0:
                    speed = distance/dt
                    saccade_speeds.append(speed)

            prev_eye_x = eye_x
            prev_time = current_time

            # -----------------------------
            # Reaction Detection
            # -----------------------------
            if state=="STIMULUS" and gaze==stimulus:

                reaction = time.time()-stimulus_time
                reaction_times.append(reaction)

                trial += 1
                correct += 1

                print(f"Trial {trial}: Stimulus={stimulus} Movement={gaze} Reaction={reaction:.2f} sec")

                experiment_data.append({
                    "Trial":trial,
                    "Stimulus":stimulus,
                    "Movement":gaze,
                    "Reaction_Time":round(reaction,2),
                    "Saccade_Speed":round(speed,2),
                    "Correct":stimulus==gaze,
                    "Timestamp":datetime.now().strftime("%H:%M:%S")
                })

                state="FIXATION"
                fixation_time=time.time()

        cv2.imshow("Eye Experiment",frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # -----------------------------
    # Results
    # -----------------------------
    print("\nExperiment Finished")

    print(f"Accuracy: {correct}/{total_trials}")

    if reaction_times:
        avg_reaction=sum(reaction_times)/len(reaction_times)
        print(f"Average Reaction Time: {avg_reaction:.2f} sec")

    if saccade_speeds:
        avg_speed=sum(saccade_speeds)/len(saccade_speeds)
        print(f"Average Saccade Speed: {avg_speed:.2f} px/sec")

    # -----------------------------
    # Save Data
    # -----------------------------
    df = pd.DataFrame(experiment_data)

    file_path="blink_gaze_data.xlsx"

    try:
        old_df=pd.read_excel(file_path)
        df=pd.concat([old_df,df],ignore_index=True)
    except:
        pass

    df.to_excel(file_path,index=False)

    print("Data saved to blink_gaze_data.xlsx")


if __name__ == "__main__":
    run_gaze_experiment()