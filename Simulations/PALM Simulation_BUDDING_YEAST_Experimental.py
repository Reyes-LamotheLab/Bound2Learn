import cv2
import numpy as npy
import random as rnd
from scipy import ndimage
import math as mt
from skimage.external import tifffile
import pandas as pd
from scipy.stats import expon
import scipy.io as sio
#NEEDS TO BE ADAPTED BASED ON ECOLI SIMULATION!!!!
#look into the grid resolution, and make displacements in nanometers
# integer vs round?
#
#incorporate changes in diffusive state. make sure the change is one per time step.

filename = 'Yeast_Exp_1s_slowD1_lowInt.tif'
pixels = 500
resolution = 10 #tens of nanometer
images = []
size_image = pixels*resolution
image = npy.zeros((size_image, size_image, 3), npy.uint16)

interval_frame = 2 # Time interval (frames)
camera_exposure = 500
dt = 5#ms
int_steps = npy.int_(camera_exposure/dt)
num_images = 80

track_steps = int_steps*num_images*interval_frame

integrate_array = npy.arange(int_steps - 1,track_steps, int_steps*interval_frame)
integrate_array_2 = npy.arange(int_steps - 1,track_steps, int_steps*interval_frame)
integrate_array_2[0] = 0
integrate_array_2[1:] = integrate_array_2[1:] - int_steps + 1
print(integrate_array_2)
#integrate_array = npy.arange(int_steps - 1,track_steps, int_steps*interval_frame)
print(integrate_array)
track_numbers = 2
num_bacteria = 400


int_fluorophore = 2000
#Intensity within exposure time
U1_single_spot = int_fluorophore/int_steps
#Intensity per single time step

out_background = 150
#Camera noise
out_back_sigma = 20
#standard deviation of camera noise
Cell_bg_intensity = 20
#Cell autofluorescence within exposure time
cell_background = 0.1#Cell_bg_intensity/int_steps
#cell_background_sigma = 0.05
mean_psf = 13
#Point spread function. I.e how much the intensity of the spot will be spread out based on convultion with gaussian filter
std_psf = 1.0

#wid_bac_mean = 70
#std_wid = 1

radius_mean = 100
std_radius = 5

Tbleach = 20000
kbleach = 1/Tbleach# frames
Tunbind = 8000
Tbind = 10000000000
kunbind = 1/Tunbind
kbind = 1/Tbind
alpha_coo = 0.4

D1 = 0.5
var_1 = 2*D1*dt

D2 = 0.05
var_2 = 2*D2*dt**alpha_coo

w1 = 0.5
w2 = 1-w1

params_array = [interval_frame, camera_exposure,cell_background, Tbleach,Tunbind,Tbind,D1,D2, int_fluorophore, w1]
params_array_num = npy.array(params_array)
dat_df = pd.DataFrame(params_array_num)
dat_df_Tr = dat_df.T
columns_nam = ['Interval(frames)', 'Camera Exposure', 'Cell Background (per resolution)', 'Tbleach', 'Bound Time', 'Search Time', 'Fast Moving D', 'Bound D', 'IntFluorophore', 'Weight D1']
dat_df_Tr.columns = columns_nam

file_nam_params = filename.replace(".tif", "_params.csv")
dat_df_Tr.to_csv(file_nam_params)
#Array for positions of bacteria
bac_cells = npy.zeros((num_bacteria, 6))

#Borders of image
positions_lower_bnd = 0
positions_upper_bnd = 5000
#Array for initial positions of tracks
track_initials = npy.zeros((track_numbers, 3, num_bacteria))
#Array for tracks
track = npy.zeros((track_steps, 3, track_numbers, num_bacteria))
#Array for state
track_D = npy.zeros((track_numbers, 1, num_bacteria))

tracks_save = npy.zeros((num_images,2, track_numbers,num_bacteria))
tracks_save_D = npy.zeros((num_images,1, track_numbers,num_bacteria))
#States for Diffusive State
D1_st = 0
D2_st = 1
#States of fluorescence
FL_st = 2
Bl_st = 3
#Probabilities for diffusive state changes
D1_F_p = expon.cdf(x=dt, scale=Tbind)
D2_F_p = expon.cdf(x=dt, scale=Tunbind)
#Bleach probability
B_p = expon.cdf(x=dt, scale=Tbleach)
#Probabilities for no transition
D1_D1_p = 1 - D1_F_p
D2_D2_p = 1 - D2_F_p
B_p_N = 1 - B_p


Trans_mat_D = npy.array([[D1_D1_p, D1_F_p], [D2_F_p, D2_D2_p]])
Trans_mat_F = npy.array([B_p_N, B_p])

weights = [w1, w2]
D = [D1_st, D2_st]

xrang_coord = npy.arange(positions_lower_bnd + 2*(radius_mean + 3*std_radius), positions_upper_bnd - 2*(radius_mean + 3*std_radius) , 2*(radius_mean + 3*std_radius))
yrang_coord = npy.arange(positions_lower_bnd + 2*(radius_mean + 3*std_radius), positions_upper_bnd - 2*(radius_mean + 3*std_radius) , 2*(radius_mean + 3*std_radius))
[xv, yv] = npy.meshgrid(xrang_coord, yrang_coord)
num_cols = npy.shape(xv)
print(num_cols)
coord_yst_x = []
coord_yst_y = []
for uu in range(num_cols[1]):
    for vv in range(num_cols[0] ):
        #x_coord_bac_sel = uu
        #y_coord_bac_sel = vv
        coord_yst_x.append(xv[0,uu])
        coord_yst_y.append(yv[vv,0])

coords_final = [coord_yst_x,coord_yst_y]
coords_final = [[row[i] for row in coords_final]
                             for i in range(len(coords_final[0]))]
coords_final = npy.array(coords_final)

for u in range(num_bacteria):
    #xv_coord = rnd.randrange (0, num_cols-1, 1)
    #yv_coord = rnd.randrange(0, num_cols-1, 1)
    center_coord_yst_x = coords_final[u,0]
    center_coord_yst_y = coords_final[u,1]

    radius_sam = int(rnd.normalvariate(radius_mean, std_radius))
    bac_cells[u, (0,1)] = [center_coord_yst_x, center_coord_yst_y]
    bac_cells[u, 2] = radius_sam
    yst_xmin = center_coord_yst_x - radius_sam
    yst_xmax = center_coord_yst_x + radius_sam
    yst_ymin = center_coord_yst_y - radius_sam
    yst_ymax = center_coord_yst_y + radius_sam
    yst_zmin = 0 - radius_sam
    yst_zmax = 0 + radius_sam #assume all cells are at the same plane
    for w in range(track_numbers):
        x_init = rnd.randrange(yst_xmin, yst_xmax)
        x_origin = center_coord_yst_x
        y_init = rnd.randrange(yst_ymin, yst_ymax)
        y_origin = center_coord_yst_y
        z_init = rnd.randrange(yst_zmin, yst_zmax)
        z_origin = 0
        track_initials[w, :, u] = [x_init, y_init, z_init]



        while (x_init-x_origin)**2 + (y_init-y_origin)**2 + (z_init-z_origin)**2 - radius_sam**2 > 0:

            x_init = rnd.randrange(yst_xmin, yst_xmax)
            y_init = rnd.randrange(yst_ymin, yst_ymax)
            z_init = rnd.randrange(yst_zmin, yst_zmax)
            track_initials[w, :, u] = [x_init, y_init, z_init]
        track[0, :, w, u] = track_initials[w, :, u]

        Diffusion_selection =  npy.random.multinomial(1, weights)
        Diffusion_selection_elem = npy.nonzero(Diffusion_selection)
        Diffuse_state = npy.sum(Diffusion_selection_elem)
        track_D[w, 0, u] = Diffuse_state

for j in range(num_bacteria):
    yst_image = cv2.circle(image, (int(bac_cells[j,0]), int(bac_cells[j, 1])), int(bac_cells[j, 2]),
                                    (0, 255, 0), -1, 8)

yst_image_final = cv2.cvtColor(yst_image, cv2.COLOR_BGR2GRAY)
#  Check if the circles are overlapping
yst_image_16_reform = yst_image_final.reshape(pixels, resolution, pixels, resolution).sum(3).sum(1)
yst_image_16 = npy.uint16(yst_image_16_reform)
non_zer_find = npy.nonzero(yst_image_16)
yst_image_16[yst_image_16!=0] = 1
filename_bin = filename.replace(".tif", "_binary.tif")
#filename_bin.replace(".tif", "binary.tif")
#print(filename_bin)
cv2.imwrite(filename_bin, yst_image_16)
print('Initialization Complete')
with tifffile.TiffWriter(filename, bigtiff=True) as tif:

    yst_array = npy.array(yst_image_final)
    max_cell_value = npy.amax(yst_array)
    index_max = npy.argwhere(yst_array == max_cell_value)
    #index_max_background = npy.argwhere(rt4 != max_cell_value)
    #rt4[index_max] = cell_background
    for m in range(len(index_max)):
        yst_array[index_max[m, 0], index_max[m, 1]] = npy.random.poisson(cell_background, 1)


    yst_array_2 = npy.array(yst_array)
    for i in range(track_steps):
        if i in integrate_array_2:
            init_time = npy.where(integrate_array_2 == i)
            init_time = init_time[0]
            print(init_time)
            int_init = integrate_array_2[init_time].item()
            int_last = integrate_array[init_time].item()
            print(int_last)
            integrate_array_3 = npy.arange(int_init,int_last + 1)

        for u in range(num_bacteria):

            yst_xmin = bac_cells[u,0] - bac_cells[u,2]
            yst_xmax = bac_cells[u,0] + bac_cells[u,2]
            yst_ymin = bac_cells[u,1] - bac_cells[u,2]
            yst_ymax = bac_cells[u,1] + bac_cells[u,2]
            radius_sam = bac_cells[u, 2]
            yst_zmin = 0 - radius_sam
            yst_zmax = 0 + radius_sam
            x_origin = bac_cells[u,0]
            y_origin = bac_cells[u,1]
            z_origin = 0

            for w in range(track_numbers):
                #print('track initials', track_initials[w, :, u])
                if i == 0:
                    spot_int = U1_single_spot
                    spot_coord = npy.zeros((1, 2))
                    spot_coord[0, (0, 1)] = [track[i, 0, w, u], track[i, 1, w, u]]
                    spot_coord = npy.int_(spot_coord)
                    gauss_sam = mean_psf#rnd.normalvariate(mean_psf, std_psf)
                    init_spot = npy.zeros((70, 70))
                    init_spot[34, 34] = spot_int
                    if npy.absolute(int(track[i, 2, w, u])) <= 15:
                        gauss_sam = mean_psf
                    elif npy.absolute(int(track[i, 2, w, u])) > 15 and npy.absolute(int(track[i, 2, w, u])) <= 30:
                        gauss_sam = mean_psf * 1.05
                    elif npy.absolute(int(track[i, 2, w, u])) > 30 and npy.absolute(int(track[i, 2, w, u])) <= 45:
                        gauss_sam = mean_psf * 1.10
                    elif npy.absolute(int(track[i, 2, w, u])) > 45 and npy.absolute(int(track[i, 2, w, u])) <= 60:
                        gauss_sam = mean_psf * 1.15
                    elif npy.absolute(int(track[i, 2, w, u])) > 60 and npy.absolute(int(track[i, 2, w, u])) <= 75:
                        gauss_sam = mean_psf * 1.20
                    elif npy.absolute(int(track[i, 2, w, u])) > 75 and npy.absolute(int(track[i, 2, w, u])) <= 90:
                        gauss_sam = mean_psf * 1.25
                    elif npy.absolute(int(track[i, 2, w, u])) > 90 and npy.absolute(int(track[i, 2, w, u])) <= 105:
                        gauss_sam = mean_psf * 1.30
                    elif npy.absolute(int(track[i, 2, w, u])) > 105 and npy.absolute(int(track[i, 2, w, u])) <= 120:
                        gauss_sam = mean_psf * 1.35
                    else:
                        print ("PSF Error")
                    gauss_filter = ndimage.gaussian_filter(init_spot, gauss_sam, truncate=8)
                    for q in range(len(gauss_filter)):
                        for v in range(len(gauss_filter)):
                            poiss_dist = npy.random.poisson(gauss_filter[q, v], 1)
                            gauss_filter[q, v] = poiss_dist
                    spot_replace = yst_array_2[spot_coord[0, 1] - 34:spot_coord[0, 1] + 36,
                                   spot_coord[0, 0] - 34:spot_coord[0, 0] + 36]
                    if npy.size(spot_replace) != npy.size(gauss_filter):
                        print("Work on your coding skills you idiot!")
                    spot_replace_2 = npy.add(spot_replace, gauss_filter)
                    yst_array_2[spot_coord[0, 1] - 34:spot_coord[0, 1] + 36, spot_coord[0, 0] - 34:spot_coord[0, 0] + 36] = \
                        spot_replace_2
                    continue
                #If molecule is bleached we don't care about it
                if track_D [w, 0, u]  == Bl_st:
                    continue

                #Determine if molecule bleaches within next time step
                Transition_BL = Trans_mat_F
                rand_multi_select_BL = npy.random.multinomial(1, Transition_BL)
                element_multi_BL = npy.nonzero(rand_multi_select_BL)
                state_select_BL = npy.sum(element_multi_BL)

                if state_select_BL + 2 == FL_st:
                #If it is fluorescence determine if molecule transitions to different diffusive state
                    if track_D[w, 0, u] == D1_st:
                        Transition_p = Trans_mat_D[D1_st, :]
                        rand_multi_select = npy.random.multinomial(1, Transition_p)
                        element_multi = npy.nonzero(rand_multi_select)
                        state_select = npy.sum(element_multi)

                        if state_select == D1_st:
                            track_D[w, 0, u] = D1_st
                            var = var_1

                        elif state_select == D2_st:
                            track_D[w, 0, u] = D2_st
                            var = var_2

                    elif track_D[w, 0, u] == D2_st:
                        Transition_p = Trans_mat_D[D2_st, :]
                        rand_multi_select = npy.random.multinomial(1, Transition_p)
                        element_multi = npy.nonzero(rand_multi_select)
                        state_select = npy.sum(element_multi)

                        if state_select == D1_st:
                            track_D[w, 0, u] = D1_st
                            var = var_1

                        elif state_select == D2_st:
                            track_D[w, 0, u] = D2_st
                            var = var_2
                #print ('variance', var)
                    x_step = rnd.normalvariate(0, mt.sqrt(var))
                    y_step = rnd.normalvariate(0, mt.sqrt(var))
                    z_step = rnd.normalvariate(0, mt.sqrt(var))
                    track[i, 0, w, u] = track[i - 1, 0, w, u] + x_step
                    track[i, 1, w, u] = track[i - 1, 1, w, u] + y_step
                    track[i, 2, w, u] = track[i - 1, 2, w, u] + z_step

                    #print('zmin', zmin, 'zmax', zmax, 'xmin', xmin, 'xmax', xmax, 'track', track)
                    while  (track[i, 0, w, u]-x_origin)**2 + (track[i, 1, w, u]-y_origin)**2 + \
                                    (track[i, 2, w, u]- z_origin)**2 - radius_sam**2 > 0:
                        x_step = rnd.normalvariate(0, mt.sqrt(var))
                        y_step = rnd.normalvariate(0, mt.sqrt(var))
                        z_step = rnd.normalvariate(0, mt.sqrt(var))
                        track[i, 0, w, u] = track[i - 1, 0, w, u] + x_step
                        track[i, 1, w, u] = track[i - 1, 1, w, u] + y_step
                        track[i, 2, w, u] = track[i - 1, 2, w, u] + z_step

                    #spot_int = rnd.gauss(U1_single_spot, int_spot_sigma)
                    if i in integrate_array_3:
                        spot_int = U1_single_spot
                        spot_coord = npy.zeros((1, 2))
                        spot_coord[0, (0, 1)] = [track[i, 0, w, u], track[i, 1, w, u]]
                        spot_coord = npy.int_(spot_coord)
                        #print(spot_coord,"spot_coord")
                        #gauss_sam = rnd.normalvariate(mean_psf, std_psf)
                        if npy.absolute(int(track[i, 2, w, u])) <= 15:
                            gauss_sam = mean_psf
                        elif npy.absolute(int(track[i, 2, w, u])) > 15 and npy.absolute(int(track[i, 2, w, u])) <= 30:
                            gauss_sam = mean_psf * 1.05
                        elif npy.absolute(int(track[i, 2, w, u])) > 30 and npy.absolute(int(track[i, 2, w, u])) <= 45:
                            gauss_sam = mean_psf * 1.10
                        elif npy.absolute(int(track[i, 2, w, u])) > 45 and npy.absolute(int(track[i, 2, w, u])) <= 60:
                            gauss_sam = mean_psf * 1.15
                        elif npy.absolute(int(track[i, 2, w, u])) > 60 and npy.absolute(int(track[i, 2, w, u])) <= 75:
                            gauss_sam = mean_psf * 1.20
                        elif npy.absolute(int(track[i, 2, w, u])) > 75 and npy.absolute(int(track[i, 2, w, u])) <= 90:
                            gauss_sam = mean_psf * 1.25
                        elif npy.absolute(int(track[i, 2, w, u])) > 90 and npy.absolute(int(track[i, 2, w, u])) <= 105:
                            gauss_sam = mean_psf * 1.30
                        elif npy.absolute(int(track[i, 2, w, u])) > 105 and npy.absolute(int(track[i, 2, w, u])) <= 120:
                            gauss_sam = mean_psf * 1.35
                        else:
                            print ("PSF Error")

                        init_spot = npy.zeros((70, 70))
                        init_spot[34, 34] = spot_int

                        gauss_filter = ndimage.gaussian_filter(init_spot, gauss_sam, truncate=8)
                       #print("sum_gauss_after", npy.sum(gauss_filter))
                        for q in range(len(gauss_filter)):
                            for v in range(len(gauss_filter)):
                                poiss_dist = npy.random.poisson(gauss_filter[q, v], 1)
                                gauss_filter[q, v] = poiss_dist
                        #print("sum_gauss_after", npy.sum(gauss_filter))
                        spot_replace = yst_array_2[spot_coord[0, 1] - 34:spot_coord[0, 1] + 36,
                                                 spot_coord[0, 0] - 34:spot_coord[0, 0] + 36]
                        #print(spot_replace,"spot_replace")
                        if npy.size(spot_replace) != npy.size(gauss_filter):
                            print("Work on your coding skills you idiot!")

                        spot_replace_2 = npy.add(spot_replace, gauss_filter)
                        #print ("gauss_filter", gauss_filter, "spot_replace_2", spot_replace_2)
                        yst_array_2[spot_coord[0, 1] - 34:spot_coord[0, 1] + 36, spot_coord[0, 0] - 34:spot_coord[0, 0] + 36] = \
                            spot_replace_2
                elif state_select_BL + 2 == Bl_st:
                    track_D[w, 0, u] = Bl_st
                    continue
        print("Time step", i)
        if i in integrate_array:
            print (i, "save I")

            yst_array_int16 = npy.uint16(yst_array_2)
            # Reformat for pixels
            yst_array_reform = yst_array_int16.reshape(pixels, resolution, pixels, resolution).sum(3).sum(1)
            # Add EMCCD camera noise
            for m in range(len(yst_array_reform)):
                for n in range(len(yst_array_reform)):
                    yst_array_reform[m, n] = yst_array_reform[m, n] + npy.random.normal(out_background, out_back_sigma,
                                                                                        1)

            tif.save(npy.uint16(yst_array_reform))

            yst_array = npy.array(yst_image_final)
            max_cell_value = npy.amax(yst_array)
            index_max = npy.argwhere(yst_array == max_cell_value)
            for m in range(len(index_max)):
                yst_array[index_max[m, 0], index_max[m, 1]] = npy.random.poisson(cell_background, 1)

            yst_array_2 = npy.array(yst_array)

            time_pt = npy.where(integrate_array == i)
            #print(time_pt)
            time_pt = time_pt[0]
            #print(time_pt)


            if time_pt == 0:
                time_step_prev = 0
            else:
                time_pt_prev = time_pt - 1
                time_step_prev = integrate_array[time_pt_prev].item()
            time_step_curr = integrate_array[time_pt].item()
            #print(integrate_array[time_pt_prev],integrate_array[time_pt])
            print(time_step_prev, time_step_curr)
            for u in range(num_bacteria):
                for w in range(track_numbers):

                    x_mean_int = npy.mean(track[time_step_prev:time_step_curr, 0, w, u])
                    y_mean_int = npy.mean(track[time_step_prev:time_step_curr, 1, w, u])
                    tracks_save[time_pt, 0, w, u] = x_mean_int
                    tracks_save[time_pt, 1, w, u] = y_mean_int
                    tracks_save_D[time_pt, 0, w, u] = track_D[w, 0, u]
                    # will use Matlab's nonzeros function remove zeros
            print('done')
file_nam_tracks = filename.replace(".tif", "_track_results.mat")
file_nam_tracks_D = filename.replace(".tif", "_track_results_D.mat")
sio.savemat(file_nam_tracks,{'tracks_info' : tracks_save})
sio.savemat(file_nam_tracks_D,{'tracks_info_D' : tracks_save_D})








