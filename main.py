import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import cv2

# Read images
print("###########################################################################################################################")
print("[LOG]: Lendo imagens da esquerda e direita...")
IL = cv2.imread('./assets/raster/esquerda.ppm')  # left image
IR = cv2.imread('./assets/raster/direita.ppm')   # right image
gray1 = cv2.cvtColor(IL, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(IR, cv2.COLOR_BGR2GRAY)
print("[LOG]: Imagens lidas com sucesso!")


# Intrinsic parameter matrix
print("[LOG]: Lendo parâmetros intrínsecos...")
fm = 403.657593  # Focal distance in pixels
cx, cy = 161.644318, 124.202080  # Principal point coordinates (pixels)
bl = 119.929  # baseline (mm)
K = np.array([[fm, 0, cx], [0, fm, cy], [0, 0, 1.0000]])  # Same for both cameras

# Extrinsic parameters
T = np.array([-bl, 0, 0])  # Translation between cameras
R = np.eye(3)  # Rotation matrix (identity matrix)


print("[LOG]: Parâmetros lidos com sucesso!")


def plane_sweep_ncc(im_l, im_r, start, steps, wid):
    print("[LOG]: Iniciando busca de disparidade com NCC...")
    m, n = im_l.shape
    mean_l, mean_r = ndimage.uniform_filter(im_l, wid), ndimage.uniform_filter(im_r, wid)
    norm_l, norm_r = im_l - mean_l, im_r - mean_r
    dmaps = np.zeros((m, n, steps))

    for displ in range(steps):
        rolled_norm_r = np.roll(norm_r, displ + start)
        s = ndimage.uniform_filter(norm_l * rolled_norm_r, wid)
        s_l = ndimage.uniform_filter(norm_l * norm_l, wid)
        s_r = ndimage.uniform_filter(rolled_norm_r * rolled_norm_r, wid)
        dmaps[:, :, displ] = s / np.sqrt(np.absolute(s_l * s_r))

    return np.argmax(dmaps, axis=2) + start

def plane_sweep_gauss(im_l, im_r, start, steps, wid):
    print("[LOG]: Iniciando busca de disparidade com Gauss...")
    m, n = im_l.shape
    mean_l, mean_r = ndimage.gaussian_filter(im_l, wid), ndimage.gaussian_filter(im_r, wid)
    norm_l, norm_r = im_l - mean_l, im_r - mean_r
    dmaps = np.zeros((m, n, steps))

    for displ in range(steps):
        rolled_norm_r = np.roll(norm_r, displ + start)
        s = ndimage.gaussian_filter(norm_l * rolled_norm_r, wid)
        s_l = ndimage.gaussian_filter(norm_l * norm_l, wid)
        s_r = ndimage.gaussian_filter(rolled_norm_r * rolled_norm_r, wid)
        dmaps[:, :, displ] = s / np.sqrt(s_l * s_r)

    return np.argmax(dmaps, axis=2) + start

# Main
print("###########################################################################################################################")
print("[LOG]: Iniciando busca de disparidade...")
im_l = np.array(Image.open('./assets/raster/esquerda.ppm').convert('L'), 'f')
im_r = np.array(Image.open('./assets/raster/direita.ppm').convert('L'), 'f')

steps, start = 28, 25
wid1, wid2 = 11, 3
res1 = plane_sweep_ncc(im_l, im_r, start, steps, wid1)
res2 = plane_sweep_gauss(im_l, im_r, start, steps, wid2)

# Calculate real depth Z in millimeters
print("[LOG]: Calculando profundidade real Z em milímetros...")
disparity = res2.astype(np.float32)
Z_real = np.where(disparity == 0, np.inf, (fm * bl) / disparity)

# Create pixel grid
print("[LOG]: Criando grade de pixels...")
u, v = np.meshgrid(np.arange(im_l.shape[1]), np.arange(im_l.shape[0]))

# Calculate real X and Y
print("[LOG]: Calculando X e Y reais...")
X_real = ((u - cx) * Z_real) / fm
Y_real = ((v - cy) * Z_real) / fm

print("[LOG]: Iniciando visualização...")
fig = plt.figure(figsize=(15, 10))

# Left and Right Images
print("[LOG]: Iniciando visualização das imagens esquerda e direita...")
ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(IL[..., ::-1])
ax1.set_title('IMAGEM ESQUERDA')

ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(IR[..., ::-1])
ax2.set_title('IMAGEM DIREITA')

# Gaussian Result
print("[LOG]: Iniciando visualização do resultado Gaussiano...")
ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(res2, cmap='gray')
ax3.set_title('RESULTADO GAUSSIANO')

# 3D Depth Map
print("[LOG]: Iniciando visualização do mapa de profundidade 3D...")
ax4 = fig.add_subplot(2, 3, (4, 5), projection='3d')
Z = np.where(res2 == 0, np.inf, 1 / res2)
X, Y = np.meshgrid(np.arange(im_l.shape[1]), np.arange(im_l.shape[0]))
mask = ~np.isinf(Z.flatten())
ax4.scatter(X.flatten()[mask][::5], Y.flatten()[mask][::5], Z.flatten()[mask][::5], c=Z.flatten()[mask][::5], cmap='Oranges', marker='.')
ax4.view_init(elev=-75, azim=-90)
ax4.set_title('MAPA DE PROFUNDIDADE 3D GAUSS')

# 3D Reconstruction
print("[LOG]: Iniciando visualização da reconstrução 3D...")
ax5 = fig.add_subplot(2, 3, (5, 6), projection='3d')
IL_rgb = cv2.cvtColor(IL, cv2.COLOR_BGR2RGB) 
colors = IL_rgb.reshape((-1, 3)) / 255.0

X_plot, Y_plot, Z_plot = X_real.flatten()[mask][::5], Y_real.flatten()[mask][::5], Z_real.flatten()[mask][::5]
color_plot = colors[mask][::5]
ax5.plot_surface(X, Y, Z, facecolors=IL_rgb/255, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
ax5.view_init(elev=-75, azim=-90)
ax5.set_title('RECONSTRUCAO 3D GAUSS')

plt.tight_layout()
plt.show()
print("###########################################################################################################################")
