#%%
import os, sys, bz2, cv2, math, time, pickle, itertools

import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from imutils import face_utils
from operator import itemgetter
from scipy.signal import convolve
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from math import pi, cos, sin, exp, sqrt
from sklearn.preprocessing import StandardScaler
from utils import decompress_pickle, compress_pickle

def get_landmarks_mp(img_dir):
    mp_face_mesh = mp.solutions.face_mesh
    landmarks = []
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        image = cv2.imread(img_dir)
        y,x,_ = image.shape
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        contors = mp_face_mesh.FACEMESH_TESSELATION
        contors = [sorted(x) for x in contors]
        contors = get_unique(contors)
        contors = sorted(contors, key=itemgetter(0,1))
        for landmark in results.multi_face_landmarks[0].landmark:
            landmarks.append([landmark.x*x, landmark.y*y, landmark.z*x])
        return landmarks, contors

def get_distances(contors, landmarks):
    distances = []
    for contor in contors:
        l1 = np.array(landmarks[contor[0]])
        l2 = np.array(landmarks[contor[1]])
        distances.append(math.dist(l1,l2))
    return distances

def calc_angle(A, B, C):
    Ax, Ay, Az = A[0]-B[0], A[1]-B[1], A[2]-B[2]
    Cx, Cy, Cz = C[0]-B[0], C[1]-B[1], C[2]-B[2]
    a = [Ay, Ax, Az]
    c = [Cy, Cx, Cz]
    cross = np.dot(a,c)
    mag_a = np.linalg.norm(a)
    mag_c = np.linalg.norm(c)
    return np.arccos(cross/(mag_a*mag_c))*180/pi

def get_angles_mp(cont_list,landmarks):
    """ Calculate all angles within the triangles formed from the given contors

    Args:
        cont_list (list): a list of 2 element list containing the number of the landmarks that are connected
        landmarks (np.array()): an array containing the coordinates of the lanmarks on the image

    Returns:
        angles (list): a list containing all angles of all triangles generated from the contor lines
    """
    angles = []
    ids = []
    cont_arr = np.array(cont_list)
    for i in range(len(landmarks)):
        pts_list = cont_arr[ (cont_arr[:,0] == i) |  (cont_arr[:,1] == i)]
        pts = pts_list[pts_list != i]
        for comb in list(itertools.combinations(pts,2)):
            if sorted(comb) in cont_list and [i,sorted(comb)] not in ids:
                ids.append([i,sorted([comb])])
                angles.append(calc_angle(landmarks[sorted(comb)[0]], landmarks[i], landmarks[sorted(comb)[1]]))
    return angles

def plot_mp_landmarks(landmarks, contors = None, annotate = False, img_dir = None):
    _, ax = plt.subplots(figsize=(15,18))
    if type(img_dir) == str:
        img = Image.open(img_dir)
        ax.imshow(img)
    elif img_dir is not None:   
        img = img_dir
        ax.imshow(img)
    else:
        ax.invert_yaxis()
    if annotate == True:
        for i, landmark in enumerate(landmarks):
            ax.annotate(str(i), (landmark[0], landmark[1]))
    if contors:
        landmarks = np.array(landmarks)
        for i, contor in enumerate(contors):
            x, y, _ = zip(*landmarks[np.array(contor)])
            ax.plot(np.array(x), np.array(y),color="gray",linewidth=0.4)

def get_unique(k):
    new_k = []
    for elem in k:
        if elem not in new_k:
            new_k.append(elem)
    return new_k

def get_norm_landmarks(img_dir, landmarks):
    eye_centers = []
    for idx in [[362, 263], [133, 33]]:
        x_center = int(np.mean([landmarks[idx[0]][0],landmarks[idx[1]][0]]))
        y_center = int(np.mean([landmarks[idx[0]][1],landmarks[idx[1]][1]]))
        z_center = (np.mean([landmarks[idx[0]][2],landmarks[idx[1]][2]]))
        eye_centers.append([x_center,y_center,z_center])
    inner_eye_distace = math.dist(eye_centers[0],eye_centers[1])
    c = 300/inner_eye_distace
    return np.array(landmarks)*c, c

def reshape_img(img_dir, c, show=False):
    img = Image.open(img_dir).convert('L')
    y,x = np.shape(img)
    img_resize = img.resize((int(x*c),int(y*c)))
    
    if show:
        plt.figure(figsize=(12,15))
        plt.imshow(img_resize)
    return np.array(img_resize)
    
def self_gabor(sigma, theta, Lambda, psi, gamma):
    """Gabor feature extraction."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 5  # Number of standard deviations
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Construct the real component of the gabor filter 
    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    
    return gb

def get_gb_fb():
    gb_fb = dict()
    Lambda = [4, 4*sqrt(2) ,8 ,8*sqrt(2) ,16]   # wavelength                        
    alpha = [4, 6, 8, 10]                       # orientation
    phi = 0                                     # Phase shift

    for l in Lambda:
        for s in [l/4,l/2,3*l/4,l]:             # scaling of filter
            for a in alpha:
                    gb_fb[f'\u03BB:{round(l,2)},  \u03C3:{s},  \u0398:{round(a,2)}'] = self_gabor(sigma=s, theta=a, Lambda=l, psi=phi, gamma=1)
    return gb_fb

def get_plot_range(landmarks_norm):
    xmax, xmin, ymax, ymin = max(landmarks_norm[:,0]).astype(int), min(landmarks_norm[:,0]).astype(int), max(landmarks_norm[:,1]).astype(int), min(landmarks_norm[:,1]).astype(int)

def main(i, img_dir, subset=None):
    
    if "DS" in img_dir:
        return 
    
    # Extract key - different for emotionet and disfa -
    main_key = int(img_dir[-9:-4])

    try:
        # Generate Shape Vector
        if subset:
            landmarks, _ = get_landmarks_mp(img_dir)
            #contors = np.load("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/src/assets/subset_contors.npy")
            contors = np.load("/zhome/08/3/117881/MastersThesis/DataProcessing/EmotionModel/src/assets/subset_contors.npy")
            landmark_idx = np.unique(contors).astype(int)
        else:
            landmarks, contors = get_landmarks_mp(img_dir)
        landmarks_norm, c = get_norm_landmarks(img_dir, landmarks)
        
        if subset:
            landmarks_norm = landmarks_norm[landmark_idx]

        pts = landmarks_norm[:,:2]
        dist = np.abs(pts[np.newaxis, :, :] - pts[:, np.newaxis, :]).min(axis=2)
        dist2 = dist[np.triu_indices(pts.shape[0], 1)].tolist()

        feat_x = np.array(dist2)                             
        
        return {main_key: feat_x}
    
    except:
        print(f'Image {main_key} could not be handled')
        return {main_key: np.nan}
        
#%%
if __name__ == "__main__":
    os.environ["GLOG_minloglevel"] ="2"
    if sys.platform == "linux":
        dir_path = "/work3/s164272/data/EmotioNetData/"
        pickles_path = "/work3/s164272/data/Features"
    else:
        dir_path = "/Users/DG/Documents/PasswordProtected/TestImg/"#"/Users/DG/Documents/PasswordProtected/EmotioNetTest/"
        pickles_path = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles"
    face_space = dict()
    misses = []

    print("Generation started....")
    # Parallel generation of face_space vectors
    dictionary_list = Parallel(n_jobs=24,verbose=10)(delayed(main)(i,f'{dir_path}{file}', subset=True) for i, file in enumerate(sorted(os.listdir(dir_path))))
    print("Generation done!!!")

    print("Dictionary combination started....")
    for d in dictionary_list:
        try:
            face_space.update(d)
        except:
            misses.append(d)


    print("Compressin bz2 pickle files...")
    print(face_space)
    #face_space = face_space.astype(np.float32)
    np.save(f"{pickles_path}/shape_space_emotioline_subset_300.npy", face_space)
    np.save(f"{pickles_path}/misses_shape_emotioline_subset_300.npy", misses)
    #compress_pickle(f"{pickles_path}/face_space_dict_disfa_large1", face_space)
    #compress_pickle(f"{pickles_path}/misses_disfa_large1", misses)
    print("All done!...")
    time.sleep(1)
    print("Well done")

# %%
