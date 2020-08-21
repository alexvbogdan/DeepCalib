import numpy as np
import cv2
import math as m
import cmath as cm
import time, glob
import my_interpol
from scipy import interpolate
import random
import pdb
from numpy.lib.scimath import sqrt as csqrt
random.seed(9001)
np.random.seed(1)


def deg2rad(deg):
    return deg*m.pi/180

def getRotationMat(roll, pitch, yaw):

    rx = np.array([1., 0., 0., 0., np.cos(deg2rad(roll)), -np.sin(deg2rad(roll)), 0., np.sin(deg2rad(roll)), np.cos(deg2rad(roll))]).reshape((3, 3))
    ry = np.array([np.cos(deg2rad(pitch)), 0., np.sin(deg2rad(pitch)), 0., 1., 0., -np.sin(deg2rad(pitch)), 0., np.cos(deg2rad(pitch))]).reshape((3, 3))
    rz = np.array([np.cos(deg2rad(yaw)), -np.sin(deg2rad(yaw)), 0., np.sin(deg2rad(yaw)), np.cos(deg2rad(yaw)), 0., 0., 0., 1.]).reshape((3, 3))

    return np.matmul(rz, np.matmul(ry, rx))

def minfocal( u0,v0,xi,xref=1,yref=1):

    fmin = np.sqrt(-(1-xi*xi)*((xref-u0)*(xref-u0) + (yref-v0)*(yref-v0)))

    return fmin * 1.0001

def diskradius(xi, f):

    return np.sqrt(-(f*f)/(1-xi*xi))



#----------------constants--------------
path_to_360_images = '/home/user/Documents/SUN360_urls_9104x4552/all_images_new/*.jpg'
list_360_image_paths = glob.glob(path_to_360_images)

H=299
W=299
u0 = W / 2.
v0 = H / 2.

grid_x, grid_y = np.meshgrid(range(W), range(H))

starttime = time.clock()
for image360_path in list_360_image_paths: #length of your filename list

    image360 = cv2.imread(image360_path)
    #image360 = cv2.imread('/home/user/Documents/SUN360_urls_9104x4552/all_images_new/gymindoor_8_.jpg')

    ImPano_W = np.shape(image360)[1]
    ImPano_H = np.shape(image360)[0]


    for i in range(1):
        x_ref = 1
        y_ref = 1
        f = random.randrange(50, 500, 10)
        xi = random.randrange(0, 120, 2)
        xi = xi / 100.

        fmin = minfocal(u0, v0, xi, x_ref, y_ref)

        # 1. Projection on the camera plane

        X_Cam = np.divide(grid_x- u0, f)
        Y_Cam = np.divide(grid_y- v0, f)

        # 2. Projection on the sphere

        AuxVal = np.multiply(X_Cam, X_Cam) + np.multiply(Y_Cam, Y_Cam)

        alpha_cam = np.real(xi + csqrt(1 + np.multiply((1 - xi * xi), AuxVal)))
        
        alpha_div = AuxVal + 1

        alpha_cam_div = np.divide(alpha_cam, alpha_div)

        X_Sph = np.multiply(X_Cam, alpha_cam_div)
        Y_Sph = np.multiply(Y_Cam, alpha_cam_div)
        Z_Sph = alpha_cam_div - xi

        # 3. Rotation of the sphere
        Rot = []
        Rot.append(((np.random.ranf() - 0.5) * 2) * 10) # roll
        Rot.append(((np.random.ranf() - 0.5) * 2) * 15) # pitch
        Rot.append(((np.random.ranf() - 0.5) * 2) * 180) # yaw

        r = np.matmul(getRotationMat(Rot[0], Rot[1], Rot[2]),
                      np.matmul(getRotationMat(0, -90, 45), getRotationMat(0, 90, 90)))

        idx1 = np.array([[0], [0], [0]])
        idx2 = np.array([[1], [1], [1]])
        idx3 = np.array([[2], [2], [2]])
        elems1 = r[:, 0]
        elems2 = r[:, 1]
        elems3 = r[:, 2]

        x1 = elems1[0] * X_Sph + elems2[0] * Y_Sph + elems3[0] * Z_Sph
        y1 = elems1[1] * X_Sph + elems2[1] * Y_Sph + elems3[1] * Z_Sph
        z1 = elems1[2] * X_Sph + elems2[2] * Y_Sph + elems3[2] * Z_Sph

        X_Sph = x1
        Y_Sph = y1
        Z_Sph = z1

        # 4. cart 2 sph
        ntheta = np.arctan2(Y_Sph, X_Sph)
        nphi = np.arctan2(Z_Sph, np.sqrt(np.multiply(X_Sph, X_Sph) + np.multiply(Y_Sph, Y_Sph)))

        pi = m.pi

        # 5. Sphere to pano
        min_theta = -pi
        max_theta = pi
        min_phi = -pi / 2.
        max_phi = pi / 2.

        min_x = 0
        max_x = ImPano_W - 1.0
        min_y = 0
        max_y = ImPano_H - 1.0

        ## for x
        a = (max_theta - min_theta) / (max_x - min_x)
        b = max_theta - a * max_x  # from y=ax+b %% -a;
        nx = (1. / a)* (ntheta - b)

        ## for y
        a = (min_phi - max_phi) / (max_y - min_y)
        b = max_phi - a * min_y  # from y=ax+b %% -a;
        ny = (1. / a)* (nphi - b)

        # 6. Final step interpolation and mapping
        im = np.array(my_interpol.interp2linear(image360,nx, ny), dtype=np.uint8)

        if f < fmin:
            r = diskradius(xi, f)
            DIM = im.shape
            ci = (np.round(DIM[0]/2), np.round(DIM[1]/2))
            xx, yy = np.meshgrid(range(DIM[0])-ci[0], range(DIM[1])-ci[1])
            mask = np.double((np.multiply(xx,xx)+np.multiply(yy,yy))<r*r)
            mask_3channel = np.stack([mask,mask,mask],axis=-1)
            im = np.array(np.multiply(im, mask_3channel),dtype=np.uint8)

        name = image360_path.split('/')[-1]
        name_list = name.split('.')

        cv2.imwrite('/home/user/Documents/test_discrete/' + name_list[0] +'_f_'+str(f)+'_d_'+str(xi)+ '.' +name_list[-1], im)


print "elapsed time ", time.clock() - starttime