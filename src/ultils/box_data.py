import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy import ndimage, misc


# Read images
IL = cv2.imread('./assets/raster/esquerda.ppm') # left image
IR = cv2.imread('./assets/raster/direita.ppm')  # right image
gray1 = cv2.cvtColor(IL, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(IR, cv2.COLOR_BGR2GRAY)

print(IL.shape)


#Explicação
# Tenho o Z do mapa de profundidade, mas nao tenho as coordenadas tridimensionais, só os vetores (x,y) do tamanho da imagem
# Atraves da formula, X = (Zx - Z0x) / Fsx, para calcular do Z

# intrinsic parameter matrix
fm = 403.657593 # Focal distantce in pixels  -> FX
cx = 161.644318 # Principal point - x-coordinate (pixels)  -> OY
cy = 124.202080 # Principal point - y-coordinate (pixels)  -> OX
bl = 119.929 # baseline (mm) -> BASE LINE
# for the right camera    
right_k = np.array([[ fm, 0, cx],[0, fm, cy],[0, 0, 1.0000]])

# for the left camera
left_k = np.array([[fm, 0, cx],[0, fm, cy],[0, 0, 1.0000]]) # K -> utilizado

# Extrinsic parameters
# Translation between cameras
T = np.array([-bl, 0, 0]) 
# Rotation
R = np.array([[ 1,0,0],[ 0,1,0],[0,0,1]])

print('Intrinsic Paramenters')
print('Left_K:\n', left_k)
print('Right_K:\n', right_k)

print('Extrinsic Paramenters')
print('R:\n', R)
print('T:\n', T)


# Show images 
fig, ax = plt.subplots(nrows=1, ncols=2)
plt.subplot(1, 2, 1)
plt.imshow(IL,cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(IR,cmap='gray')
plt.show(block=False)

plt.show()

